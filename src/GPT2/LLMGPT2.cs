//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace LLM.ILGPU;

public class LLMGPT2 : ILLM
{
    private readonly Accelerator _accelerator;
    private readonly Context _context;

    public ITokenizer Tokenizer { get; }
    public Vocab Vocab { get; }

    private readonly List<ILayer> _network;
    private readonly LossManager _lossManager;
    public int MaxSeqLen { get; }

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _inputIdsBuffer;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _targetIdsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _inferProbsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _gradNormBuffer;

    // ── Пинированные буферы (page-locked) для быстрых копий на iGPU ──
    private readonly int[] _pinnedInputCpu;
    private readonly int[] _pinnedTargetCpu;
    private readonly float[] _pinnedProbsCpu;

    // ── Кэш токенов для инференса ────────────────────────────────────
    private readonly int[] _inferTokenCache;

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

        _network.Add(new EmbeddingLayer(
            _accelerator, vocabSize, embeddingDim, maxSeqLen));
        _network.Add(new PositionalEmbeddingLayer(
            _accelerator, maxSeqLen, embeddingDim));
        for (int i = 0; i < numLayers; i++)
            _network.Add(new TransformerBlockPreNorm(
                _accelerator, embeddingDim, numHeads, hiddenDim, maxSeqLen));
        _network.Add(new LayerNormLayer(
            _accelerator, embeddingDim, maxSeqLen));
        _network.Add(new LinearLayer(
            _accelerator, embeddingDim, vocabSize, true, maxSeqLen));

        _lossManager = new LossManager(_accelerator, vocabSize, MaxSeqLen);

        _inputIdsBuffer = _accelerator.Allocate1D<int>(MaxSeqLen);
        _targetIdsBuffer = _accelerator.Allocate1D<int>(MaxSeqLen);
        _inferProbsBuffer = _accelerator.Allocate1D<float>(
            (long)MaxSeqLen * vocabSize);
        _gradNormBuffer = _accelerator.Allocate1D<float>(1);

        // GC.AllocateArray pinned:true — page-locked память,
        // DMA-копии CPU↔GPU быстрее для shared memory iGPU
        _pinnedInputCpu = GC.AllocateArray<int>(MaxSeqLen, pinned: true);
        _pinnedTargetCpu = GC.AllocateArray<int>(MaxSeqLen, pinned: true);
        _pinnedProbsCpu = GC.AllocateArray<float>(vocabSize, pinned: true);
        _inferTokenCache = new int[maxSeqLen];
    }

    // ═══════════════════════════════════════════════════════
    // ИНФЕРЕНС
    // ═══════════════════════════════════════════════════════

    public string Predict(string userInput, float temperature = 0.7f)
    {
        string prompt = Tokenizer.FormatDialogue(userInput);
        return PredictRaw(prompt, temperature);
    }

    public string PredictRaw(string prompt, float temperature = 0.7f)
    {
        var tokenized = Tokenizer.Encode(prompt, addBos: true, addEos: false);
        if (tokenized.Count == 0) return string.Empty;

        if (tokenized.Count > MaxSeqLen - 10)
            tokenized = tokenized.Take(MaxSeqLen - 10).ToList();
        if (tokenized.Count >= MaxSeqLen) return string.Empty;

        int vocabSize = Tokenizer.VocabSize;
        var outputTokens = new List<int>(MaxSeqLen);

        // Копируем промпт в кэш
        int currentLen = tokenized.Count;
        for (int i = 0; i < currentLen; i++)
            _inferTokenCache[i] = tokenized[i];

        for (int step = 0; step < MaxSeqLen; step++)
        {
            if (currentLen >= MaxSeqLen - 1) break;

            CopyTokensToGpu(_inferTokenCache, currentLen, _inputIdsBuffer);

            var logits = RunForward(currentLen);

            MatrixOps.Softmax(_accelerator, logits,
                _inferProbsBuffer, currentLen, vocabSize);

            int nextToken = SampleWithTemperature(
                _inferProbsBuffer, currentLen, vocabSize, temperature);

            if (nextToken == Tokenizer.EosId) break;
            if (nextToken == Tokenizer.PadId) continue;

            outputTokens.Add(nextToken);

            if (currentLen < _inferTokenCache.Length)
                _inferTokenCache[currentLen] = nextToken;
            currentLen++;
        }

        return Tokenizer.Decode(outputTokens);
    }

    // ═══════════════════════════════════════════════════════
    // ОБУЧЕНИЕ
    // ═══════════════════════════════════════════════════════

    /// <summary>
    /// Обычное обучение
    /// Обновляет веса после каждого примера
    /// </summary>
    public float TrainStep(List<int> tokens, float lr)
    {
        if (tokens.Count < 2) return 0f;

        int seqLen = tokens.Count - 1;
        if (seqLen > MaxSeqLen)
        {
            Console.WriteLine(
                $"⚠ TrainStep: seqLen={seqLen} > MaxSeqLen={MaxSeqLen}, обрезаем");
            seqLen = MaxSeqLen;
            tokens = tokens.Take(seqLen + 1).ToList();
        }

        // Заполняем пинированные буферы без new[]
        for (int i = 0; i < seqLen; i++)
        {
            _pinnedInputCpu[i] = tokens[i];
            _pinnedTargetCpu[i] = tokens[i + 1];
        }

        // CopyFromCPU(T[]) — копирует весь массив,
        // поэтому используем SubView + вспомогательный метод
        CopyTokensToGpu(_pinnedInputCpu, seqLen, _inputIdsBuffer);
        CopyTokensToGpu(_pinnedTargetCpu, seqLen, _targetIdsBuffer);

        var logits = RunForward(seqLen);

        var (probsView, loss) = _lossManager.ComputeSoftmaxAndLoss(
            logits,
            _targetIdsBuffer.View.SubView(0, seqLen),
            seqLen);

        var gradsView = _lossManager.ComputeGradients(
            probsView,
            _targetIdsBuffer.View.SubView(0, seqLen),
            seqLen);

        MatrixOps.ClipGradients(
            _accelerator, gradsView, 1.0f, _gradNormBuffer);

        var gradView = gradsView;
        for (int i = _network.Count - 1; i >= 0; i--)
            gradView = _network[i].Backward(gradView, lr);

        _accelerator.Synchronize();
        return loss;
    }

    /// <summary>
    /// Батч-обучение: gradient accumulation по N примерам,
    /// один Synchronize на весь батч.
    /// Оптимально для gfx1150: batchSize = 4..16.
    /// </summary>
    public float TrainBatch(List<List<int>> batch, float lr)
    {
        if (batch.Count == 0) return 0f;

        // ── Фильтрация валидных примеров ──────────────────────────
        var valid = new List<(List<int> tokens, int seqLen)>();
        foreach (var tokens in batch)
        {
            if (tokens.Count < 2) continue;
            int seqLen = Math.Min(tokens.Count - 1, MaxSeqLen);
            valid.Add((tokens, seqLen));
        }

        if (valid.Count == 0) return 0f;

        int N = valid.Count;
        float scaledLr = lr / N;  // ← ключевое: lr/N за шаг = lr суммарно
        float totalLoss = 0f;
        int padId = Tokenizer.PadId;

        for (int s = 0; s < N; s++)
        {
            var (tokens, seqLen) = valid[s];

            // ── Заполняем буферы ──────────────────────────────────
            for (int i = 0; i < seqLen; i++)
            {
                _pinnedInputCpu[i] = tokens[i];
                _pinnedTargetCpu[i] = tokens[i + 1];
            }
            // Явно заполняем остаток PAD — буфер мог содержать старые данные
            for (int i = seqLen; i < MaxSeqLen; i++)
            {
                _pinnedInputCpu[i] = padId;
                _pinnedTargetCpu[i] = padId;
            }

            CopyTokensToGpu(_pinnedInputCpu, seqLen, _inputIdsBuffer);
            CopyTokensToGpu(_pinnedTargetCpu, seqLen, _targetIdsBuffer);

            // ── Forward ───────────────────────────────────────────
            var logits = RunForward(seqLen);

            // ── Loss ──────────────────────────────────────────────
            var (probsView, loss) = _lossManager.ComputeSoftmaxAndLoss(
                logits,
                _targetIdsBuffer.View.SubView(0, seqLen),
                seqLen);

            totalLoss += loss;

            // ── Градиент ──────────────────────────────────────────
            var gradsView = _lossManager.ComputeGradients(
                probsView,
                _targetIdsBuffer.View.SubView(0, seqLen),
                seqLen);

            MatrixOps.ClipGradients(_accelerator, gradsView, 1.0f, _gradNormBuffer);

            // ── Backward с lr/N ───────────────────────────────────
            // Математически эквивалентно gradient accumulation при SGD:
            //   Δw = -lr * (1/N) * Σ grad_i
            //      = -lr/N * grad_1 + (-lr/N * grad_2) + ...
            var gradView = gradsView;
            for (int i = _network.Count - 1; i >= 0; i--)
                gradView = _network[i].Backward(gradView, scaledLr);
        }

        // Один Synchronize на весь батч
        _accelerator.Synchronize();

        return totalLoss / N;
    }

    // ═══════════════════════════════════════════════════════
    // Вспомогательный метод копирования токенов на GPU
    // ═══════════════════════════════════════════════════════

    /// <summary>
    /// ILGPU SubView.CopyFromCPU(T[]) требует массив ТОЧНО нужной длины.
    /// Этот метод копирует первые <paramref name="count"/> элементов
    /// из <paramref name="src"/> в GPU-буфер без аллокации нового массива,
    /// используя пре-аллоцированный _pinnedInputCpu как промежуточник
    /// (src уже является одним из пинированных буферов).
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private void CopyTokensToGpu(
        int[] src,
        int count,
        MemoryBuffer1D<int, Stride1D.Dense> dst)
    {
        // SubView(0, count).CopyFromCPU(array) — array должен быть >= count.
        // src (пинированный) имеет длину MaxSeqLen >= count — ок.
        dst.View.SubView(0, count).CopyFromCPU(src);
    }

    // ═══════════════════════════════════════════════════════
    // Forward pass
    // ═══════════════════════════════════════════════════════

    private ArrayView1D<float, Stride1D.Dense> RunForward(int seqLen)
    {
        if (_network.Count == 0)
            throw new InvalidOperationException("Сеть пуста");

        var embLayer = _network[0] as EmbeddingLayer
            ?? throw new InvalidOperationException(
                "Первый слой должен быть EmbeddingLayer");

        ArrayView1D<float, Stride1D.Dense> x =
            embLayer.Forward(
                _inputIdsBuffer.View.SubView(0, seqLen), seqLen);

        for (int i = 1; i < _network.Count; i++)
            x = _network[i].Forward(x, seqLen);

        return x;
    }

    // ═══════════════════════════════════════════════════════
    // Sampling
    // ═══════════════════════════════════════════════════════

    private int SampleWithTemperature(
        MemoryBuffer1D<float, Stride1D.Dense> probs,
        int seqLen, int vocabSize, float temperature)
    {
        int offset = (seqLen - 1) * vocabSize;

        // CopyToCPU(T[]) — копирует в массив длиной >= SubView.Length
        // _pinnedProbsCpu имеет длину vocabSize — точное совпадение
        probs.View.SubView(offset, vocabSize).CopyToCPU(_pinnedProbsCpu);

        if (temperature <= 0.01f)
        {
            int maxIdx = 0;
            float maxVal = _pinnedProbsCpu[0];
            for (int i = 1; i < vocabSize; i++)
            {
                if (_pinnedProbsCpu[i] > maxVal)
                {
                    maxVal = _pinnedProbsCpu[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        float invTemp = 1.0f / temperature;
        double sum = 0.0;

        for (int i = 0; i < vocabSize; i++)
        {
            double logP = Math.Log(Math.Max(_pinnedProbsCpu[i], 1e-10f));
            _pinnedProbsCpu[i] = (float)Math.Exp(logP * invTemp);
            sum += _pinnedProbsCpu[i];
        }

        if (sum <= 0.0) sum = 1.0;
        float invSum = (float)(1.0 / sum);
        for (int i = 0; i < vocabSize; i++)
            _pinnedProbsCpu[i] *= invSum;

        float r = (float)Random.Shared.NextDouble();
        float cumulative = 0f;
        for (int i = 0; i < vocabSize; i++)
        {
            cumulative += _pinnedProbsCpu[i];
            if (r <= cumulative) return i;
        }
        return vocabSize - 1;
    }

    // ═══════════════════════════════════════════════════════
    // Утилиты
    // ═══════════════════════════════════════════════════════

    public int TotalParameters() =>
        _network.Sum(l => l.Parameters());

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
            _inputIdsBuffer.Dispose();
            _targetIdsBuffer.Dispose();
            _inferProbsBuffer.Dispose();
            _gradNormBuffer.Dispose();

            _disposed = true;
        }
    }

    // ═══════════════════════════════════════════════════════
    // WarmUp
    // ═══════════════════════════════════════════════════════

    public void WarmUpInternal()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        Console.WriteLine($"  Компиляция GPU kernels {MaxSeqLen}...");

        RunWarmUpPass(MaxSeqLen, lr: 0f);
        _accelerator.Synchronize();
        Console.WriteLine($"  JIT done: {sw.ElapsedMilliseconds}ms");

        Console.WriteLine("  Прогрев GPU Command Queue...");
        const int queueWarmCount = 3;
        for (int i = 0; i < queueWarmCount; i++)
        {
            RunWarmUpPass(MaxSeqLen, lr: 0f);
            _accelerator.Synchronize();
            Console.Write(
                $"  [{i + 1}/{queueWarmCount}] {sw.ElapsedMilliseconds}ms\r");
        }
        Console.WriteLine();

        sw.Stop();
        Console.WriteLine(
            $"  Прогрев завершён за {sw.ElapsedMilliseconds}ms");
    }

    private void RunWarmUpPass(int seqLen, float lr)
    {
        // Используем пинированные буферы — без new[]
        Array.Clear(_pinnedInputCpu, 0, seqLen);
        Array.Clear(_pinnedTargetCpu, 0, seqLen);

        CopyTokensToGpu(_pinnedInputCpu, seqLen, _inputIdsBuffer);
        CopyTokensToGpu(_pinnedTargetCpu, seqLen, _targetIdsBuffer);

        var logits = RunForward(seqLen);

        var (probsView, _) = _lossManager.ComputeSoftmaxAndLoss(
            logits,
            _targetIdsBuffer.View.SubView(0, seqLen),
            seqLen);

        var gradsView = _lossManager.ComputeGradients(
            probsView,
            _targetIdsBuffer.View.SubView(0, seqLen),
            seqLen);

        var gradView = gradsView;
        for (int i = _network.Count - 1; i >= 0; i--)
            gradView = _network[i].Backward(gradView, lr);
    }
}