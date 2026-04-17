//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;

namespace LLM.ILGPU;

public static class LinearKernels
{
    public static void LinearForwardKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int seqLen,
        SpecializedValue<int> inFeatures,
        SpecializedValue<int> outFeatures)
    {
        int i = index.X;
        int j = index.Y;
        if (i >= seqLen || j >= outFeatures) return;

        float sum = 0;
        for (int k = 0; k < inFeatures; k++)
            sum += input[i * inFeatures + k] * weight[k * outFeatures + j];

        output[i * outFeatures + j] = sum + bias[j];
    }

    public static void LinearWeightGradKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradWeight,
        int seqLen,
        SpecializedValue<int> inFeatures,
        SpecializedValue<int> outFeatures)
    {
        int k = index.X;
        int j = index.Y;
        if (k >= inFeatures || j >= outFeatures) return;

        float sum = 0;
        for (int i = 0; i < seqLen; i++)
            sum += input[i * inFeatures + k] * gradOutput[i * outFeatures + j];

        gradWeight[k * outFeatures + j] = sum;
    }

    // ✅ Убрано деление на seqLen — масштаб должен совпадать с weight grad
    public static void LinearBiasGradKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradBias,
        int seqLen,
        SpecializedValue<int> outFeatures)
    {
        int j = index;
        float sum = 0;
        for (int i = 0; i < seqLen; i++)
            sum += gradOutput[i * outFeatures + j];

        gradBias[j] = sum;
    }

    public static void LinearInputGradKernel(
        Index2D index,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> weight,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        int seqLen,
        SpecializedValue<int> inFeatures,
        SpecializedValue<int> outFeatures)
    {
        int i = index.X;
        int k = index.Y;
        if (i >= seqLen || k >= inFeatures) return;

        float sum = 0;
        for (int j = 0; j < outFeatures; j++)
            sum += gradOutput[i * outFeatures + j] * weight[k * outFeatures + j];

        gradInput[i * inFeatures + k] = sum;
    }
}

public class LinearLayer : ILayer
{
    private readonly Accelerator _accelerator;
    internal readonly int _inFeatures;
    internal readonly int _outFeatures;
    private readonly int _maxSeqLen;

    internal MemoryBuffer1D<float, Stride1D.Dense> _weight;
    internal MemoryBuffer1D<float, Stride1D.Dense> _bias;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradWeight;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradBias;
    private MemoryBuffer1D<float, Stride1D.Dense> _outputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradInputBuffer;

    private ArrayView1D<float, Stride1D.Dense> _cachedInputView;
    private int _cachedSeqLen;

    private readonly Action<Index2D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _forwardKernel;

    private readonly Action<Index2D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _weightGradKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _biasGradKernel;

    private readonly Action<Index2D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _inputGradKernel;

    private readonly Adam _optimizerWeight;
    private readonly Adam _optimizerBias;
    private bool _disposed;

    public LinearLayer(Accelerator accelerator, int inFeatures, int outFeatures,
        bool useBias = true, int maxSeqLen = 80)
    {
        _accelerator = accelerator;
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;
        _maxSeqLen = maxSeqLen;

        int weightSize = inFeatures * outFeatures;
        _weight = accelerator.Allocate1D<float>(weightSize);
        _bias = accelerator.Allocate1D<float>(outFeatures);
        _gradWeight = accelerator.Allocate1D<float>(weightSize);
        _gradBias = accelerator.Allocate1D<float>(outFeatures);

        int maxOutputSize = maxSeqLen * outFeatures;
        int maxInputSize = maxSeqLen * inFeatures;
        _outputBuffer = accelerator.Allocate1D<float>(maxOutputSize);
        _gradInputBuffer = accelerator.Allocate1D<float>(maxInputSize);

        float std = (float)Math.Sqrt(2.0 / inFeatures);
        MatrixOps.RandomNormalInit(accelerator, _weight, std);
        MatrixOps.ZeroInit(accelerator, _bias);

        _forwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            LinearKernels.LinearForwardKernel);

        _weightGradKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            LinearKernels.LinearWeightGradKernel);

        _biasGradKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            LinearKernels.LinearBiasGradKernel);

        _inputGradKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            LinearKernels.LinearInputGradKernel);

        _optimizerWeight = new Adam(accelerator, weightSize);
        _optimizerBias = new Adam(accelerator, outFeatures);
    }

    public string LayerType => "Linear";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        if (seqLen <= 0)
            throw new ArgumentException($"seqLen должен быть > 0, получено {seqLen}");

        int inputSize = seqLen * _inFeatures;
        _cachedSeqLen = seqLen;
        _cachedInputView = input.SubView(0, inputSize);

        int outputSize = seqLen * _outFeatures;
        var outputView = _outputBuffer.View.SubView(0, outputSize);

        _forwardKernel(new Index2D(seqLen, _outFeatures),
            _cachedInputView, _weight.View, _bias.View, outputView,
            seqLen,
            SpecializedValue.New(_inFeatures),
            SpecializedValue.New(_outFeatures));

        return outputView;
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr)
    {
        if (_cachedSeqLen <= 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        int seqLen = _cachedSeqLen;

        _weightGradKernel(new Index2D(_inFeatures, _outFeatures),
            _cachedInputView, grads, _gradWeight.View,
            seqLen,
            SpecializedValue.New(_inFeatures),
            SpecializedValue.New(_outFeatures));

        _biasGradKernel(new Index1D(_outFeatures),
            grads, _gradBias.View,
            seqLen, SpecializedValue.New(_outFeatures));

        int inputSize = seqLen * _inFeatures;
        var gradInputView = _gradInputBuffer.View.SubView(0, inputSize);

        _inputGradKernel(new Index2D(seqLen, _inFeatures),
            grads, _weight.View, gradInputView,
            seqLen,
            SpecializedValue.New(_inFeatures),
            SpecializedValue.New(_outFeatures));

        _optimizerWeight.Step(_weight, _gradWeight, lr);
        _optimizerBias.Step(_bias, _gradBias, lr);

        _cachedSeqLen = 0;
        return gradInputView;
    }

    public int Parameters() => (int)_weight.Length + (int)_bias.Length;

    public void Dispose()
    {
        if (!_disposed)
        {
            _weight.Dispose(); _bias.Dispose();
            _gradWeight.Dispose(); _gradBias.Dispose();
            _outputBuffer.Dispose(); _gradInputBuffer.Dispose();
            _optimizerWeight.Dispose(); _optimizerBias.Dispose();
            _disposed = true;
        }
    }
}