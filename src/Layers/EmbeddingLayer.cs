//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class EmbeddingKernels
{
    public static void EmbeddingForwardKernel(
        Index2D index,
        ArrayView1D<int, Stride1D.Dense> inputIds,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> output,
        SpecializedValue<int> embeddingDim)
    {
        int seqIdx = index.X;
        int dimIdx = index.Y;
        if (seqIdx >= inputIds.IntLength || dimIdx >= embeddingDim) return;

        int tokenId = inputIds[seqIdx];
        output[seqIdx * embeddingDim + dimIdx] =
            weights[tokenId * embeddingDim + dimIdx];
    }

    public static void EmbeddingBackwardKernel(
        Index2D index,
        ArrayView1D<int, Stride1D.Dense> inputIds,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradWeights,
        SpecializedValue<int> embeddingDim)
    {
        int seqIdx = index.X;
        int dimIdx = index.Y;
        if (seqIdx >= inputIds.IntLength || dimIdx >= embeddingDim) return;

        int tokenId = inputIds[seqIdx];
        float grad = gradOutput[seqIdx * embeddingDim + dimIdx];
        Atomic.Add(ref gradWeights[tokenId * embeddingDim + dimIdx], grad);
    }

    public static void AddPositionalEncodingKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> data,
        int seqLen,
        SpecializedValue<int> embeddingDim)
    {
        int pos = index.X;
        int d = index.Y;
        if (pos >= seqLen || d >= embeddingDim) return;

        float divTerm = XMath.Pow(10000.0f,
            (2.0f * (d / 2)) / (float)embeddingDim);
        float angle = pos / divTerm;
        float peValue = (d % 2 == 0) ? XMath.Sin(angle) : XMath.Cos(angle);
        data[pos * embeddingDim + d] += peValue;
    }
}

public class EmbeddingLayer : ILayer
{
    private readonly Accelerator _accelerator;
    private readonly int _vocabSize;
    public int EmbeddingDim { get; }
    public int MaxSeqLen { get; }

    public MemoryBuffer1D<float, Stride1D.Dense> TokenWeights;
    public MemoryBuffer1D<float, Stride1D.Dense> GradTokenWeights;

    private MemoryBuffer1D<int, Stride1D.Dense> _inputIdsBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _outputBuffer;
    private MemoryBuffer1D<int, Stride1D.Dense> _cachedInputIdsBuffer;
    private int _cachedSeqLen;

    private readonly Action<Index2D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _forwardKernel;

    private readonly Action<Index2D, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _backwardKernel;

    private readonly Action<Index2D, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _posEncodingKernel;

    private readonly Adam _tokenOptimizer;
    private bool _disposed;

    public EmbeddingLayer(Accelerator accelerator, int vocabSize,
        int embeddingDim, int maxSeqLen)
    {
        _accelerator = accelerator;
        _vocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        MaxSeqLen = maxSeqLen;

        TokenWeights = accelerator.Allocate1D<float>(vocabSize * embeddingDim);
        GradTokenWeights = accelerator.Allocate1D<float>(vocabSize * embeddingDim);
        _inputIdsBuffer = accelerator.Allocate1D<int>(maxSeqLen);
        _outputBuffer = accelerator.Allocate1D<float>(maxSeqLen * embeddingDim);
        _cachedInputIdsBuffer = accelerator.Allocate1D<int>(maxSeqLen);

        InitializeWeights();

        _forwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, SpecializedValue<int>>(
            EmbeddingKernels.EmbeddingForwardKernel);

        _backwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, SpecializedValue<int>>(
            EmbeddingKernels.EmbeddingBackwardKernel);

        _posEncodingKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            EmbeddingKernels.AddPositionalEncodingKernel);

        _tokenOptimizer = new Adam(accelerator, (int)TokenWeights.Length);
    }

    private void InitializeWeights()
    {
        float std = (float)Math.Sqrt(2.0 / (_vocabSize + EmbeddingDim));
        var tokenHost = new float[TokenWeights.Length];
        var rnd = new Random(42);
        for (int i = 0; i < tokenHost.Length; i++)
        {
            float u1 = (float)rnd.NextDouble();
            float u2 = (float)rnd.NextDouble();
            float z = (float)Math.Sqrt(-2.0 * Math.Log(u1)) *
                      (float)Math.Cos(2.0 * Math.PI * u2);
            tokenHost[i] = std * z;
        }
        TokenWeights.CopyFromCPU(tokenHost);
        GradTokenWeights.MemSetToZero();
    }

    public string LayerType => "Embedding";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<int, Stride1D.Dense> inputIdsView, int seqLen)
    {
        _cachedSeqLen = seqLen;
        _cachedInputIdsBuffer.View.SubView(0, seqLen)
            .CopyFrom(inputIdsView.SubView(0, seqLen));

        var currentOutputView = _outputBuffer.View.SubView(0,
            seqLen * EmbeddingDim);

        _forwardKernel(
            new Index2D(seqLen, EmbeddingDim),
            inputIdsView, TokenWeights.View, currentOutputView,
            SpecializedValue.New(EmbeddingDim));

        _posEncodingKernel(
            new Index2D(seqLen, EmbeddingDim),
            currentOutputView, seqLen,
            SpecializedValue.New(EmbeddingDim));

        return currentOutputView;
    }

    public ArrayView1D<float, Stride1D.Dense> Forward(int[] tokenIds, int seqLen)
    {
        _inputIdsBuffer.View.SubView(0, seqLen).CopyFromCPU(tokenIds);
        return Forward(_inputIdsBuffer.View.SubView(0, seqLen), seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        // Fallback: конвертация float → int (медленно, избегать)
        var inputHost = new float[input.Length];
        input.CopyToCPU(inputHost);
        var tokenIds = inputHost.Select(x => (int)x).ToArray();
        return Forward(tokenIds, seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr)
    {
        if (_cachedSeqLen == 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        _backwardKernel(
            new Index2D(_cachedSeqLen, EmbeddingDim),
            _cachedInputIdsBuffer.View.SubView(0, _cachedSeqLen),
            grads, GradTokenWeights.View,
            SpecializedValue.New(EmbeddingDim));

        _tokenOptimizer.Step(TokenWeights, GradTokenWeights, lr);
        return grads;
    }

    public int Parameters() => (int)TokenWeights.Length;

    public void Dispose()
    {
        if (!_disposed)
        {
            TokenWeights.Dispose();
            GradTokenWeights.Dispose();
            _inputIdsBuffer.Dispose();
            _outputBuffer.Dispose();
            _cachedInputIdsBuffer.Dispose();
            _tokenOptimizer.Dispose();
            _disposed = true;
        }
    }
}