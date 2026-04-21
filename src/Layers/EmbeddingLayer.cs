//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using System.Linq;

namespace LLM.ILGPU;

public static class EmbeddingKernels
{
    public static void EmbeddingForwardKernel(
        Index2D index,
        ArrayView1D<int, Stride1D.Dense> inputIds,
        ArrayView1D<float, Stride1D.Dense> weights,
        ArrayView1D<float, Stride1D.Dense> output,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<int> vocabSize)
    {
        int seqIdx = index.X;
        int dimIdx = index.Y;
        if (seqIdx >= inputIds.IntLength || dimIdx >= embeddingDim) return;

        int tokenId = inputIds[seqIdx];
        if (tokenId < 0 || tokenId >= vocabSize)
        {
            output[seqIdx * embeddingDim + dimIdx] = 0.0f;
            return;
        }
        output[seqIdx * embeddingDim + dimIdx] =
            weights[tokenId * embeddingDim + dimIdx];
    }

    public static void EmbeddingBackwardKernel(
        Index2D index,
        ArrayView1D<int, Stride1D.Dense> inputIds,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradWeights,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<int> vocabSize)
    {
        int seqIdx = index.X;
        int dimIdx = index.Y;
        if (seqIdx >= inputIds.IntLength || dimIdx >= embeddingDim) return;

        int tokenId = inputIds[seqIdx];
        if (tokenId < 0 || tokenId >= vocabSize) return;

        float grad = gradOutput[seqIdx * embeddingDim + dimIdx];
        Atomic.Add(ref gradWeights[tokenId * embeddingDim + dimIdx], grad);
    }

    public static void BuildPositionalEncodingKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> peTable,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<int> maxSeqLen)
    {
        int pos = index.X;
        int d = index.Y;
        if (pos >= maxSeqLen || d >= embeddingDim) return;

        int i = d / 2;
        float exponent = (2.0f * i) / (float)(int)embeddingDim;
        float divTerm = XMath.Exp(exponent * XMath.Log(10000.0f));
        float angle = (float)pos / divTerm;

        peTable[pos * embeddingDim + d] = (d % 2 == 0)
            ? XMath.Sin(angle)
            : XMath.Cos(angle);
    }

    public static void ApplyPositionalEncodingKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> data,
        ArrayView1D<float, Stride1D.Dense> peTable,
        SpecializedValue<int> embeddingDim)
    {
        int pos = index.X;
        int d = index.Y;
        if (d >= embeddingDim) return;

        int linearIdx = pos * embeddingDim + d;
        data[linearIdx] += peTable[linearIdx];
    }

    // ── НОВЫЙ kernel: float→int cast на GPU (убирает CPU round-trip) ──
    // Используется в Forward(float[], int) вместо CopyToCPU+cast+CopyFromCPU
    public static void FloatToIntKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<int, Stride1D.Dense> output)
    {
        output[index] = (int)input[index];
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

    private readonly MemoryBuffer1D<float, Stride1D.Dense> _peTable;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _inputIdsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _outputBuffer;

    // ── НОВЫЙ буфер: для GPU-side float→int конвертации ──────────
    // Избегаем CPU round-trip в Forward(float[], int)
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _floatToIntBuffer;

    private ArrayView1D<int, Stride1D.Dense> _cachedInputIdsView;
    private int _cachedSeqLen;

    // ── Пре-аллоцированный CPU буфер для InitializeWeights ───────
    // Переиспользуем вместо new float[big] каждый раз
    private readonly float[] _initWeightsBuf;

    private readonly Action<Index2D,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>,
        SpecializedValue<int>> _forwardKernel;

    private readonly Action<Index2D,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>,
        SpecializedValue<int>> _backwardKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>,
        SpecializedValue<int>> _buildPeKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _applyPeKernel;

    // ── НОВЫЙ kernel: float→int на GPU ───────────────────────────
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>> _floatToIntKernel;

    private readonly Adam _tokenOptimizer;
    private bool _disposed;

    public EmbeddingLayer(
        Accelerator accelerator,
        int vocabSize,
        int embeddingDim,
        int maxSeqLen,
        float weightDecay = 0.0f)
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _accelerator = accelerator;
        _vocabSize = vocabSize;
        EmbeddingDim = embeddingDim;
        MaxSeqLen = maxSeqLen;

        TokenWeights = accelerator.Allocate1D<float>(vocabSize * embeddingDim);
        GradTokenWeights = accelerator.Allocate1D<float>(vocabSize * embeddingDim);
        _peTable = accelerator.Allocate1D<float>(maxSeqLen * embeddingDim);
        _inputIdsBuffer = accelerator.Allocate1D<int>(maxSeqLen);
        _outputBuffer = accelerator.Allocate1D<float>(maxSeqLen * embeddingDim);

        // Буфер для GPU float→int (размер maxSeqLen — не vocabSize)
        _floatToIntBuffer = accelerator.Allocate1D<int>(maxSeqLen);

        // Пре-аллоцируем CPU буфер для инициализации весов
        _initWeightsBuf = GC.AllocateArray<float>(
            (int)TokenWeights.Length, pinned: true);

        InitializeWeights();
        GradTokenWeights.MemSetToZero();

        _forwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>,
            SpecializedValue<int>>(EmbeddingKernels.EmbeddingForwardKernel);

        _backwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>,
            SpecializedValue<int>>(EmbeddingKernels.EmbeddingBackwardKernel);

        _buildPeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>,
            SpecializedValue<int>>(EmbeddingKernels.BuildPositionalEncodingKernel);

        _applyPeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(EmbeddingKernels.ApplyPositionalEncodingKernel);

        _floatToIntKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>>(EmbeddingKernels.FloatToIntKernel);

        BuildPositionalEncodingTable();

        _tokenOptimizer = new Adam(
            accelerator,
            (int)TokenWeights.Length,
            weightDecay: weightDecay);
    }

    private void BuildPositionalEncodingTable()
    {
        _buildPeKernel(
            new Index2D(MaxSeqLen, EmbeddingDim),
            _peTable.View,
            SpecializedValue.New(EmbeddingDim),
            SpecializedValue.New(MaxSeqLen));
        _accelerator.Synchronize();
    }

    private void InitializeWeights()
    {
        // Используем пре-аллоцированный пинированный буфер — нет GC давления
        float limit = (float)Math.Sqrt(1.0 / EmbeddingDim);
        var rnd = new Random(42);

        for (int i = 0; i < _initWeightsBuf.Length; i++)
        {
            double u1 = rnd.NextDouble() + 1e-10;
            double u2 = rnd.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1))
                      * Math.Cos(2.0 * Math.PI * u2);
            _initWeightsBuf[i] = (float)(limit * z);
        }
        TokenWeights.CopyFromCPU(_initWeightsBuf);
    }

    public string LayerType => "Embedding";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<int, Stride1D.Dense> inputIdsView, int seqLen)
    {
        if (seqLen <= 0 || seqLen > MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen),
                $"seqLen={seqLen} выходит за пределы [1, {MaxSeqLen}]");

        _cachedSeqLen = seqLen;
        _cachedInputIdsView = inputIdsView.SubView(0, seqLen);

        var outputView = _outputBuffer.View.SubView(0, seqLen * EmbeddingDim);

        _forwardKernel(
            new Index2D(seqLen, EmbeddingDim),
            inputIdsView,
            TokenWeights.View,
            outputView,
            SpecializedValue.New(EmbeddingDim),
            SpecializedValue.New(_vocabSize));

        _applyPeKernel(
            new Index2D(seqLen, EmbeddingDim),
            outputView,
            _peTable.View,
            SpecializedValue.New(EmbeddingDim));

        return outputView;
    }

    public ArrayView1D<float, Stride1D.Dense> Forward(int[] tokenIds, int seqLen)
    {
        if (tokenIds == null)
            throw new ArgumentNullException(nameof(tokenIds));
        if (seqLen > tokenIds.Length)
            throw new ArgumentException(
                $"seqLen={seqLen} > tokenIds.Length={tokenIds.Length}");
        if (seqLen > MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen));

        // CopyFromCPU требует массив >= seqLen
        _inputIdsBuffer.View.SubView(0, seqLen).CopyFromCPU(tokenIds);
        return Forward(_inputIdsBuffer.View.SubView(0, seqLen), seqLen);
    }

    /// <summary>
    /// Fallback: float[] → int[] конвертация НА GPU (без CPU round-trip).
    /// Используем FloatToIntKernel вместо CopyToCPU+cast+CopyFromCPU.
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        // Конвертируем float→int прямо на GPU — нет CopyToCPU
        var intView = _floatToIntBuffer.View.SubView(0, seqLen);
        _floatToIntKernel(seqLen, input.SubView(0, seqLen), intView);
        return Forward(intView, seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr)
    {
        if (_cachedSeqLen == 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        GradTokenWeights.MemSetToZero();

        _backwardKernel(
            new Index2D(_cachedSeqLen, EmbeddingDim),
            _cachedInputIdsView,
            grads,
            GradTokenWeights.View,
            SpecializedValue.New(EmbeddingDim),
            SpecializedValue.New(_vocabSize));

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
            _peTable.Dispose();
            _inputIdsBuffer.Dispose();
            _outputBuffer.Dispose();
            _floatToIntBuffer.Dispose();
            _tokenOptimizer.Dispose();
            _disposed = true;
        }
    }
}