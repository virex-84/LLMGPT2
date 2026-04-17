//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class LayerNormKernels
{
    public static void ComputeMeanKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> mean,
        SpecializedValue<int> embeddingDim)
    {
        float sum = 0;
        for (int j = 0; j < embeddingDim; j++)
            sum += input[row * embeddingDim + j];
        mean[row] = sum / embeddingDim;
    }

    public static void ComputeVarianceKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> mean,
        ArrayView1D<float, Stride1D.Dense> variance,
        SpecializedValue<int> embeddingDim,
        SpecializedValue<float> epsilon)
    {
        float m = mean[row];
        float sumSq = 0;
        for (int j = 0; j < embeddingDim; j++)
        {
            float diff = input[row * embeddingDim + j] - m;
            sumSq += diff * diff;
        }
        variance[row] = sumSq / embeddingDim + epsilon;
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
        float std = XMath.Sqrt(variance[i] + epsilon);
        float normalized = (input[i * embeddingDim + j] - m) / std;
        output[i * embeddingDim + j] = gamma[j] * normalized + beta[j];
    }

    /// <summary>
    /// ✅ Один поток на строку — нет race condition на shared буферах.
    /// </summary>
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
        float m = mean[i];
        float v = variance[i];
        float stdInv = 1.0f / XMath.Sqrt(v + epsilon);

        // Проход 1: суммы для строки
        float sum1 = 0;
        float sum2 = 0;
        for (int j = 0; j < dim; j++)
        {
            float gO = gradOutput[i * dim + j];
            float norm = (input[i * dim + j] - m) * stdInv;
            float gO_gamma = gO * gamma[j];
            sum1 += gO_gamma;
            sum2 += gO_gamma * norm;
        }

        // Проход 2: gradInput + аккумуляция gradGamma/gradBeta
        for (int j = 0; j < dim; j++)
        {
            float gO = gradOutput[i * dim + j];
            float norm = (input[i * dim + j] - m) * stdInv;
            float gO_gamma = gO * gamma[j];

            gradInput[i * dim + j] = (stdInv / dim) *
                (dim * gO_gamma - sum1 - norm * sum2);

            Atomic.Add(ref gradGamma[j], gO * norm);
            Atomic.Add(ref gradBeta[j], gO);
        }
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

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _computeMeanKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>, SpecializedValue<float>> _computeVarianceKernel;

    private readonly Action<Index2D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<float>> _normalizeKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
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

        var gammaHost = Enumerable.Repeat(1.0f, embeddingDim).ToArray();
        _gamma.CopyFromCPU(gammaHost);
        MatrixOps.ZeroInit(accelerator, _beta);

        _computeMeanKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(LayerNormKernels.ComputeMeanKernel);

        _computeVarianceKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>, SpecializedValue<float>>(
            LayerNormKernels.ComputeVarianceKernel);

        _normalizeKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<float>>(
            LayerNormKernels.LayerNormNormalizeKernel);

        _backwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
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

        _inputBuffer.View.SubView(0, inputSize)
            .CopyFrom(input.SubView(0, inputSize));

        _computeMeanKernel(seqLen,
            _inputBuffer.View.SubView(0, inputSize),
            _meanBuffer.View.SubView(0, seqLen),
            SpecializedValue.New(_embeddingDim));

        _computeVarianceKernel(seqLen,
            _inputBuffer.View.SubView(0, inputSize),
            _meanBuffer.View.SubView(0, seqLen),
            _varianceBuffer.View.SubView(0, seqLen),
            SpecializedValue.New(_embeddingDim),
            SpecializedValue.New(_epsilon));

        _normalizeKernel(new Index2D(seqLen, _embeddingDim),
            _inputBuffer.View.SubView(0, inputSize),
            _gamma.View, _beta.View,
            _outputBuffer.View.SubView(0, inputSize),
            _meanBuffer.View.SubView(0, seqLen),
            _varianceBuffer.View.SubView(0, seqLen),
            seqLen,
            SpecializedValue.New(_embeddingDim),
            SpecializedValue.New(_epsilon));

        return _outputBuffer.View.SubView(0, inputSize);
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

        _backwardKernel(seqLen,
            grads,
            _inputBuffer.View.SubView(0, inputSize),
            _gamma.View,
            _meanBuffer.View.SubView(0, seqLen),
            _varianceBuffer.View.SubView(0, seqLen),
            _gradInputBuffer.View.SubView(0, inputSize),
            _gradGamma.View, _gradBeta.View,
            SpecializedValue.New(_embeddingDim),
            SpecializedValue.New(_epsilon),
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