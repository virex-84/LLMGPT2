//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class LayerNormKernels
{
    // ── ComputeMeanKernel — без изменений, уже оптимизирован x4 ──
    public static void ComputeMeanKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> mean,
        SpecializedValue<int> embeddingDim)
    {
        int dim = embeddingDim;
        int rowBase = row * dim;

        float sum0 = 0f, sum1 = 0f, sum2 = 0f, sum3 = 0f;
        int j = 0;
        int limit4 = dim - (dim % 4);
        for (; j < limit4; j += 4)
        {
            sum0 += input[rowBase + j];
            sum1 += input[rowBase + j + 1];
            sum2 += input[rowBase + j + 2];
            sum3 += input[rowBase + j + 3];
        }
        float sum = sum0 + sum1 + sum2 + sum3;
        for (; j < dim; j++) sum += input[rowBase + j];
        mean[row] = sum / dim;
    }

    public static void ComputeVarianceKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<float> epsilon)
    {
        int dim = embeddingDim;
        int rowBase = row * dim;
        float m = mean[row];

        float sq0 = 0f, sq1 = 0f, sq2 = 0f, sq3 = 0f;
        int j = 0;
        int limit4 = dim - (dim % 4);
        for (; j < limit4; j += 4)
        {
            float d0 = input[rowBase + j] - m;
            float d1 = input[rowBase + j + 1] - m;
            float d2 = input[rowBase + j + 2] - m;
            float d3 = input[rowBase + j + 3] - m;
            sq0 += d0 * d0; sq1 += d1 * d1;
            sq2 += d2 * d2; sq3 += d3 * d3;
        }
        float sumSq = sq0 + sq1 + sq2 + sq3;
        for (; j < dim; j++) { float d = input[rowBase + j] - m; sumSq += d * d; }
        variance[row] = sumSq / dim + epsilon;
    }

    // ── НОВЫЙ: объединённый Mean+Variance за один проход ─────────
    // Алгоритм Welford — численно стабильный, один проход по данным.
    // Для gfx1150 критично: 2x меньше чтений из global memory.
    public static void ComputeMeanVarianceWelfordKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<float> epsilon)
    {
        int dim = embeddingDim;
        int rowBase = row * dim;

        // Алгоритм Welford: один проход, численно стабилен
        float m = 0f; // текущее среднее
        float m2 = 0f; // сумма квадратов отклонений
        int count = 0;

        int j = 0;
        int limit4 = dim - (dim % 4);
        for (; j < limit4; j += 4)
        {
            // Развёртка x4 с обновлением Welford
            float x0 = input[rowBase + j];
            count++; float delta0 = x0 - m; m += delta0 / count;
            m2 += delta0 * (x0 - m);

            float x1 = input[rowBase + j + 1];
            count++; float delta1 = x1 - m; m += delta1 / count;
            m2 += delta1 * (x1 - m);

            float x2 = input[rowBase + j + 2];
            count++; float delta2 = x2 - m; m += delta2 / count;
            m2 += delta2 * (x2 - m);

            float x3 = input[rowBase + j + 3];
            count++; float delta3 = x3 - m; m += delta3 / count;
            m2 += delta3 * (x3 - m);
        }
        for (; j < dim; j++)
        {
            float x = input[rowBase + j];
            count++; float delta = x - m; m += delta / count;
            m2 += delta * (x - m);
        }

        mean[row] = m;
        variance[row] = m2 / dim + epsilon; // epsilon уже добавлен
    }

    public static void LayerNormNormalizeKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        int seqLen,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<float> epsilon)
    {
        int i = index.X;
        int j = index.Y;
        if (i >= seqLen || j >= embeddingDim) return;

        float m = mean[i];
        float stdInv = 1.0f / XMath.Sqrt(variance[i]);
        float norm = (input[i * embeddingDim + j] - m) * stdInv;
        output[i * embeddingDim + j] = gamma[j] * norm + beta[j];
    }

    public static void LayerNormBackwardKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        ArrayView1D<float, Stride1D.Dense> gradGamma,
        ArrayView1D<float, Stride1D.Dense> gradBeta,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<float> epsilon,
        int seqLen)
    {
        int i = row;
        if (i >= seqLen) return;

        int dim = embeddingDim;
        int rowBase = i * dim;
        float m = mean[i];
        float stdInv = 1.0f / XMath.Sqrt(variance[i]);

        float s1_0 = 0f, s1_1 = 0f, s1_2 = 0f, s1_3 = 0f;
        float s2_0 = 0f, s2_1 = 0f, s2_2 = 0f, s2_3 = 0f;
        int j = 0;
        int limit4 = dim - (dim % 4);

        for (; j < limit4; j += 4)
        {
            float gO0 = gradOutput[rowBase + j];
            float gO1 = gradOutput[rowBase + j + 1];
            float gO2 = gradOutput[rowBase + j + 2];
            float gO3 = gradOutput[rowBase + j + 3];
            float n0 = (input[rowBase + j] - m) * stdInv;
            float n1 = (input[rowBase + j + 1] - m) * stdInv;
            float n2 = (input[rowBase + j + 2] - m) * stdInv;
            float n3 = (input[rowBase + j + 3] - m) * stdInv;
            float gg0 = gO0 * gamma[j];
            float gg1 = gO1 * gamma[j + 1];
            float gg2 = gO2 * gamma[j + 2];
            float gg3 = gO3 * gamma[j + 3];
            s1_0 += gg0; s1_1 += gg1; s1_2 += gg2; s1_3 += gg3;
            s2_0 += gg0 * n0; s2_1 += gg1 * n1;
            s2_2 += gg2 * n2; s2_3 += gg3 * n3;
        }

        float sum1 = s1_0 + s1_1 + s1_2 + s1_3;
        float sum2 = s2_0 + s2_1 + s2_2 + s2_3;
        for (; j < dim; j++)
        {
            float gO = gradOutput[rowBase + j];
            float n = (input[rowBase + j] - m) * stdInv;
            float gg = gO * gamma[j];
            sum1 += gg; sum2 += gg * n;
        }

        float scale = stdInv / dim;
        j = 0;
        for (; j < limit4; j += 4)
        {
            float gO0 = gradOutput[rowBase + j];
            float gO1 = gradOutput[rowBase + j + 1];
            float gO2 = gradOutput[rowBase + j + 2];
            float gO3 = gradOutput[rowBase + j + 3];
            float n0 = (input[rowBase + j] - m) * stdInv;
            float n1 = (input[rowBase + j + 1] - m) * stdInv;
            float n2 = (input[rowBase + j + 2] - m) * stdInv;
            float n3 = (input[rowBase + j + 3] - m) * stdInv;
            float gg0 = gO0 * gamma[j];
            float gg1 = gO1 * gamma[j + 1];
            float gg2 = gO2 * gamma[j + 2];
            float gg3 = gO3 * gamma[j + 3];
            gradInput[rowBase + j] = scale * (dim * gg0 - sum1 - n0 * sum2);
            gradInput[rowBase + j + 1] = scale * (dim * gg1 - sum1 - n1 * sum2);
            gradInput[rowBase + j + 2] = scale * (dim * gg2 - sum1 - n2 * sum2);
            gradInput[rowBase + j + 3] = scale * (dim * gg3 - sum1 - n3 * sum2);
            Atomic.Add(ref gradGamma[j], gO0 * n0);
            Atomic.Add(ref gradGamma[j + 1], gO1 * n1);
            Atomic.Add(ref gradGamma[j + 2], gO2 * n2);
            Atomic.Add(ref gradGamma[j + 3], gO3 * n3);
            Atomic.Add(ref gradBeta[j], gO0);
            Atomic.Add(ref gradBeta[j + 1], gO1);
            Atomic.Add(ref gradBeta[j + 2], gO2);
            Atomic.Add(ref gradBeta[j + 3], gO3);
        }
        for (; j < dim; j++)
        {
            float gO = gradOutput[rowBase + j];
            float n = (input[rowBase + j] - m) * stdInv;
            float gg = gO * gamma[j];
            gradInput[rowBase + j] = scale * (dim * gg - sum1 - n * sum2);
            Atomic.Add(ref gradGamma[j], gO * n);
            Atomic.Add(ref gradBeta[j], gO);
        }
    }

    // ── НОВЫЙ: объединённый Normalize+CopyInput за один проход ───
    // Сохраняет input в inputCache и нормализует — 2 операции за 1 read.
    // Устраняет _inputBuffer.CopyFrom в LayerNormLayer.Forward.
    public static void NormalizeWithCacheKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gamma,
        ArrayView1D<float, Stride1D.Dense> beta,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> inputCache, // сохраняем для backward
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        int seqLen,
        SpecializedValue<int> embeddingDim)
    {
        int i = index.X;
        int j = index.Y;
        if (i >= seqLen || j >= embeddingDim) return;

        int linearIdx = i * embeddingDim + j;
        float val = input[linearIdx];
        inputCache[linearIdx] = val;          // сохраняем для backward (заменяет CopyFrom)

        float m = mean[i];
        float stdInv = 1.0f / XMath.Sqrt(variance[i]);
        float norm = (val - m) * stdInv;
        output[linearIdx] = gamma[j] * norm + beta[j];
    }
}

public class LayerNormLayer : ILayer
{
    private readonly Accelerator _accelerator;
    public readonly int _embeddingDim;
    private readonly int _maxSeqLen;
    private readonly float _epsilon = 1e-5f;

    public MemoryBuffer1D<float, Stride1D.Dense> _gamma;
    public MemoryBuffer1D<float, Stride1D.Dense> _beta;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradGamma;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradBeta;
    private MemoryBuffer1D<float, Stride1D.Dense> _inputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _outputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _meanBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _varianceBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradInputBuffer;

    private int _cachedSeqLen;

    // ── Пре-аллоцированный gamma буфер (убираем LINQ Repeat) ─────
    private readonly float[] _gammaInitBuf;

    // Kernels
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _computeMeanKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>, SpecializedValue<float>> _computeVarianceKernel;

    // ── НОВЫЙ: Welford (Mean+Variance за 1 проход) ───────────────
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>, SpecializedValue<float>> _welfordKernel;

    // ── НОВЫЙ: Normalize+CacheInput за 1 проход ──────────────────
    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _normalizeWithCacheKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>, SpecializedValue<float>, int> _backwardKernel;

    private readonly Adam _optimizerGamma;
    private readonly Adam _optimizerBeta;
    private bool _disposed;

    public LayerNormLayer(Accelerator accelerator, int embeddingDim,
        int maxSeqLen = 80)
    {
        _accelerator = accelerator;
        _embeddingDim = embeddingDim;
        _maxSeqLen = maxSeqLen;

        _gamma = accelerator.Allocate1D<float>(embeddingDim);
        _beta = accelerator.Allocate1D<float>(embeddingDim);
        _gradGamma = accelerator.Allocate1D<float>(embeddingDim);
        _gradBeta = accelerator.Allocate1D<float>(embeddingDim);

        int maxInputSize = maxSeqLen * embeddingDim;
        _inputBuffer = accelerator.Allocate1D<float>(maxInputSize);
        _outputBuffer = accelerator.Allocate1D<float>(maxInputSize);
        _meanBuffer = accelerator.Allocate1D<float>(maxSeqLen);
        _varianceBuffer = accelerator.Allocate1D<float>(maxSeqLen);
        _gradInputBuffer = accelerator.Allocate1D<float>(maxInputSize);

        // ── Инициализация gamma без LINQ ──────────────────────────
        // Оригинал: Enumerable.Repeat(1.0f, embeddingDim).ToArray() — LINQ аллокация
        // Новый: пинированный массив + Array.Fill
        _gammaInitBuf = GC.AllocateArray<float>(embeddingDim, pinned: true);
        Array.Fill(_gammaInitBuf, 1.0f);
        _gamma.CopyFromCPU(_gammaInitBuf);
        MatrixOps.ZeroInit(accelerator, _beta);

        _computeMeanKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(LayerNormKernels.ComputeMeanKernel);

        _computeVarianceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>, SpecializedValue<float>>(
            LayerNormKernels.ComputeVarianceKernel);

        // Welford: заменяет ComputeMean + ComputeVariance (2 kernel → 1)
        _welfordKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>, SpecializedValue<float>>(
            LayerNormKernels.ComputeMeanVarianceWelfordKernel);

        // NormalizeWithCache: заменяет CopyFrom + Normalize (2 ops → 1)
        _normalizeWithCacheKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            LayerNormKernels.NormalizeWithCacheKernel);

        _backwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>, SpecializedValue<float>, int>(
            LayerNormKernels.LayerNormBackwardKernel);

        _optimizerGamma = new Adam(accelerator, embeddingDim);
        _optimizerBeta = new Adam(accelerator, embeddingDim);
    }

    public string LayerType => "LayerNorm";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        int inputSize = seqLen * _embeddingDim;
        _cachedSeqLen = seqLen;

        var inputView = input.SubView(0, inputSize);
        var cachedInput = _inputBuffer.View.SubView(0, inputSize);
        var meanView = _meanBuffer.View.SubView(0, seqLen);
        var varView = _varianceBuffer.View.SubView(0, seqLen);
        var outputView = _outputBuffer.View.SubView(0, inputSize);

        // ── ОПТИМИЗАЦИЯ 1: Welford вместо двух kernel-ов ──────────
        // Оригинал: 2 dispatch (ComputeMean + ComputeVariance) = 2x чтение
        // Новый:    1 dispatch (Welford) = 1x чтение
        _welfordKernel(
            seqLen, inputView, meanView, varView,
            SpecializedValue.New(_embeddingDim),
            SpecializedValue.New(_epsilon));

        // ── ОПТИМИЗАЦИЯ 2: NormalizeWithCache вместо CopyFrom+Normalize ──
        // Оригинал: _inputBuffer.CopyFrom(input) + _normalizeKernel = 2 GPU ops
        // Новый:    NormalizeWithCacheKernel = 1 GPU op (читаем input 1 раз)
        _normalizeWithCacheKernel(
            new Index2D(seqLen, _embeddingDim),
            inputView,
            _gamma.View,
            _beta.View,
            outputView,
            cachedInput,   // записываем input в кэш внутри kernel
            meanView,
            varView,
            seqLen,
            SpecializedValue.New(_embeddingDim));

        return outputView;
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr)
    {
        if (_cachedSeqLen <= 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        int seqLen = _cachedSeqLen;
        int inputSize = seqLen * _embeddingDim;

        _gradGamma.MemSetToZero();
        _gradBeta.MemSetToZero();

        _backwardKernel(
            seqLen,
            grads,
            _inputBuffer.View.SubView(0, inputSize),
            _gamma.View,
            _meanBuffer.View.SubView(0, seqLen),
            _varianceBuffer.View.SubView(0, seqLen),
            _gradInputBuffer.View.SubView(0, inputSize),
            _gradGamma.View,
            _gradBeta.View,
            SpecializedValue.New(_embeddingDim),
            SpecializedValue.New(0f),
            seqLen);

        _optimizerGamma.Step(_gamma, _gradGamma, lr);
        _optimizerBeta.Step(_beta, _gradBeta, lr);

        return _gradInputBuffer.View.SubView(0, inputSize);
    }

    public int Parameters() => (int)_gamma.Length + (int)_beta.Length;

    public void Dispose()
    {
        if (!_disposed)
        {
            _gamma.Dispose(); _beta.Dispose();
            _gradGamma.Dispose(); _gradBeta.Dispose();
            _inputBuffer.Dispose(); _outputBuffer.Dispose();
            _meanBuffer.Dispose(); _varianceBuffer.Dispose();
            _gradInputBuffer.Dispose();
            _optimizerGamma.Dispose(); _optimizerBeta.Dispose();
            _disposed = true;
        }
    }
}