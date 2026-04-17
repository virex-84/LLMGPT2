//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace LLM.ILGPU;

/// <summary>
/// LLMGPT2 с исправленным TrainStep и Predict для корректного GGUF экспорта.
/// </summary>
public class LLMGPT2 : ILLM
{
    private readonly Accelerator _accelerator;
    private readonly Context _context;
    public ITokenizer Tokenizer { get; }
    public Vocab Vocab { get; }
    private readonly List<ILayer> _network;
    private readonly LossManager _lossManager;
    public int MaxSeqLen { get; }

    private MemoryBuffer1D<int, Stride1D.Dense>? _inputIdsBuffer;
    private MemoryBuffer1D<int, Stride1D.Dense>? _targetIdsBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _inferProbsBuffer;
    private MemoryBuffer1D<int, Stride1D.Dense>? _inferTokensBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense>? _gradNormBuffer;
    private bool _disposed;

    public LLMGPT2(
        Context gpuContext,
        ITokenizer tokenizer,
        int embeddingDim,
        int hiddenDim,
        int numHeads,
        int numLayers,
        int maxSeqLen)
    {
        _context = gpuContext;
        _accelerator = gpuContext.Accelerator;
        Tokenizer = tokenizer;
        Vocab = tokenizer.ToVocab();
        MaxSeqLen = maxSeqLen;

        int vocabSize = tokenizer.VocabSize;
        _network = new List<ILayer>();

        // Архитектура GPT-2
        _network.Add(new EmbeddingLayer(_accelerator, vocabSize, embeddingDim, maxSeqLen));
        _network.Add(new PositionalEmbeddingLayer(_accelerator, maxSeqLen, embeddingDim));
        for (int i = 0; i < numLayers; i++)
            _network.Add(new TransformerBlockPreNorm(
                _accelerator, embeddingDim, numHeads, hiddenDim, maxSeqLen));
        _network.Add(new LayerNormLayer(_accelerator, embeddingDim, maxSeqLen));
        _network.Add(new LinearLayer(_accelerator, embeddingDim, vocabSize, true, maxSeqLen));

        _lossManager = new LossManager(_accelerator, vocabSize, MaxSeqLen);
        _inputIdsBuffer = _accelerator.Allocate1D<int>(MaxSeqLen);
        _targetIdsBuffer = _accelerator.Allocate1D<int>(MaxSeqLen);
        _inferProbsBuffer = _accelerator.Allocate1D<float>(MaxSeqLen * vocabSize);
        _inferTokensBuffer = _accelerator.Allocate1D<int>(MaxSeqLen);
        _gradNormBuffer = _accelerator.Allocate1D<float>(1);

        Console.WriteLine($"  Tokenizer: {tokenizer.GetType().Name} (vocab={vocabSize})");
        Console.WriteLine($"  Embedding: {embeddingDim}, Hidden: {hiddenDim}");
        Console.WriteLine($"  Heads: {numHeads}, Layers: {numLayers}");
        Console.WriteLine($"  MaxSeqLen: {maxSeqLen}");
        Console.WriteLine($"  Параметров: {TotalParameters():N0}");
    }

    // ═══════════════════════════════════════════════════════
    // ИНФЕРЕНС
    // ═══════════════════════════════════════════════════════

    public string Predict(string userInput, float temperature = 0.7f)
    {
        // СТРОГИЙ формат — совпадает с chat_template в GGUF
        string prompt = Tokenizer.FormatDialogue(userInput);
        return PredictRaw(prompt, temperature);
    }

    public string PredictRaw(string prompt, float temperature = 0.7f)
    {
        //для инференса addBos:true, addEos:false
        var tokenized = Tokenizer.Encode(prompt, addBos:true, addEos:false);
        var outputTokens = new List<int>();

        if (tokenized.Count > MaxSeqLen - 10)
            tokenized = tokenized.Take(MaxSeqLen - 10).ToList();
        if (tokenized.Count == 0 || tokenized.Count >= MaxSeqLen)
            return string.Empty;

        int vocabSize = Tokenizer.VocabSize;

        for (int step = 0; step < MaxSeqLen - tokenized.Count; step++)
        {
            if (tokenized.Count >= MaxSeqLen - 1) break;

            int seqLen = tokenized.Count;
            _inputIdsBuffer!.View.SubView(0, seqLen)
                            .CopyFromCPU(tokenized.ToArray());

            var layerInput = RunForward(seqLen);

            MatrixOps.Softmax(_accelerator, layerInput,
                              _inferProbsBuffer!, seqLen, vocabSize);

            int nextToken = SampleWithTemperature(
                _inferProbsBuffer!, seqLen, vocabSize, temperature);

            // Стоп-условия
            if (nextToken == Tokenizer.EosId) break;
            if (nextToken == Tokenizer.PadId) continue;

            outputTokens.Add(nextToken);
            tokenized.Add(nextToken);
        }

        return Tokenizer.Decode(outputTokens);
    }

    public string PredictWithLimit(string userInput, float temperature = 0.7f,
                                   int? maxSeqLimit = null)
        => Predict(userInput, temperature);

    // ═══════════════════════════════════════════════════════
    // ОБУЧЕНИЕ
    // ═══════════════════════════════════════════════════════

    public float TrainStep(List<int> tokens, float lr)
    {
        if (tokens.Count < 2) return 0;

        int seqLen = tokens.Count - 1;
        if (seqLen > MaxSeqLen)
        {
            Console.WriteLine($"⚠ TrainStep: seqLen={seqLen} > MaxSeqLen={MaxSeqLen}");
            return 0;
        }

        var inputIds = tokens.Take(seqLen).ToArray();
        var targetIds = tokens.Skip(1).ToArray();

        _inputIdsBuffer!.View.SubView(0, seqLen).CopyFromCPU(inputIds);
        _targetIdsBuffer!.View.SubView(0, seqLen).CopyFromCPU(targetIds);

        // Forward
        var currentView = RunForward(seqLen);

        // Loss
        var (probsView, loss) = _lossManager.ComputeSoftmaxAndLoss(
            currentView,
            _targetIdsBuffer!.View.SubView(0, seqLen),
            seqLen);

        var gradsView = _lossManager.ComputeGradients(
            probsView,
            _targetIdsBuffer!.View.SubView(0, seqLen),
            seqLen);

        MatrixOps.ClipGradients(_accelerator, gradsView, 5.0f /*GradientClipMaxNorm*/, _gradNormBuffer!);

        // Backward
        ArrayView1D<float, Stride1D.Dense> gradView = gradsView;
        for (int i = _network.Count - 1; i >= 0; i--)
            gradView = _network[i].Backward(gradView, lr);

        _accelerator.Synchronize();
        return loss;
    }

    // ═══════════════════════════════════════════════════════
    // Forward pass
    // ═══════════════════════════════════════════════════════

    private ArrayView1D<float, Stride1D.Dense> RunForward(int seqLen)
    {
        ArrayView1D<float, Stride1D.Dense> x = default;

        for (int li = 0; li < _network.Count; li++)
        {
            var layer = _network[li];
            if (li == 0 && layer is EmbeddingLayer emb)
                x = emb.Forward(_inputIdsBuffer!.View.SubView(0, seqLen), seqLen);
            else if (li == 1 && layer is PositionalEmbeddingLayer posEmb)
                x = posEmb.Forward(x, seqLen);
            else if (layer is TransformerBlockPreNorm block)
                x = block.Forward(x, seqLen);
            else if (layer is LayerNormLayer ln)
                x = ln.Forward(x, seqLen);
            else if (layer is LinearLayer linear)
                x = linear.Forward(x, seqLen);
            else
                x = layer.Forward(x, seqLen);
        }

        return x;
    }

    private int SampleWithTemperature(
        MemoryBuffer1D<float, Stride1D.Dense> probs,
        int seqLen, int vocabSize, float temperature)
    {
        int offset = (seqLen - 1) * vocabSize;
        var lastProbs = new float[vocabSize];
        probs.View.SubView(offset, vocabSize).CopyToCPU(lastProbs);

        if (temperature <= 0.01f)
        {
            int maxIdx = 0;
            for (int i = 1; i < vocabSize; i++)
                if (lastProbs[i] > lastProbs[maxIdx]) maxIdx = i;
            return maxIdx;
        }

        float maxLogit = lastProbs.Max();
        double sum = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            lastProbs[i] = (float)Math.Exp(
                (Math.Log(lastProbs[i] + 1e-10f) - Math.Log(maxLogit + 1e-10f))
                / temperature);
            sum += lastProbs[i];
        }
        for (int i = 0; i < vocabSize; i++)
            lastProbs[i] /= (float)sum;

        float r = (float)Random.Shared.NextDouble();
        float cumulative = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            cumulative += lastProbs[i];
            if (r <= cumulative) return i;
        }
        return vocabSize - 1;
    }

    public int TotalParameters() => _network.Sum(l => l.Parameters());

    public string NetworkDescription() =>
        string.Join(" → ", _network.Select(l => l.LayerType));

    public List<ILayer> GetLayers() => _network;

    public void Dispose()
    {
        if (!_disposed)
        {
            foreach (var l in _network)
                if (l is IDisposable d) d.Dispose();
            _lossManager.Dispose();
            _inputIdsBuffer?.Dispose();
            _targetIdsBuffer?.Dispose();
            _inferProbsBuffer?.Dispose();
            _inferTokensBuffer?.Dispose();
            _gradNormBuffer?.Dispose();
            _disposed = true;
        }
    }
}