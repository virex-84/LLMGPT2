//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class LinearKernels
{
    // ─────────────────────────────────────────────────────────────
    // Forward: output[i,j] = sum_k(input[i,k] * weight[k,j]) + bias[j]
    // Разворот x4 по внутреннему циклу (inFeatures)
    // gfx1150: доступ к weight[k,j] — stride по outFeatures (coalesced по j)
    // ─────────────────────────────────────────────────────────────
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

        int inF = inFeatures;
        int outF = outFeatures;

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int k = 0;
        int limit4 = inF - (inF % 4);

        for (; k < limit4; k += 4)
        {
            // input[i, k..k+3] — последовательный доступ (хорошо)
            // weight[k..k+3, j] — stride=outF (для разных j в warp: coalesced)
            s0 += input[i * inF + k] * weight[k * outF + j];
            s1 += input[i * inF + k + 1] * weight[(k + 1) * outF + j];
            s2 += input[i * inF + k + 2] * weight[(k + 2) * outF + j];
            s3 += input[i * inF + k + 3] * weight[(k + 3) * outF + j];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; k < inF; k++)
            sum += input[i * inF + k] * weight[k * outF + j];

        output[i * outF + j] = sum + bias[j];
    }

    // ─────────────────────────────────────────────────────────────
    // WeightGrad: gradWeight[k,j] = sum_i(input[i,k] * gradOutput[i,j])
    // Разворот x4 по seqLen
    // ─────────────────────────────────────────────────────────────
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

        int inF = inFeatures;
        int outF = outFeatures;

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int i = 0;
        int limit4 = seqLen - (seqLen % 4);

        for (; i < limit4; i += 4)
        {
            s0 += input[i * inF + k] * gradOutput[i * outF + j];
            s1 += input[(i + 1) * inF + k] * gradOutput[(i + 1) * outF + j];
            s2 += input[(i + 2) * inF + k] * gradOutput[(i + 2) * outF + j];
            s3 += input[(i + 3) * inF + k] * gradOutput[(i + 3) * outF + j];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; i < seqLen; i++)
            sum += input[i * inF + k] * gradOutput[i * outF + j];

        gradWeight[k * outF + j] = sum;
    }

    // ─────────────────────────────────────────────────────────────
    // BiasGrad: gradBias[j] = sum_i(gradOutput[i,j])
    // Разворот x4 по seqLen
    // ─────────────────────────────────────────────────────────────
    public static void LinearBiasGradKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradBias,
        int seqLen,
        SpecializedValue<int> outFeatures)
    {
        int j = index;
        int outF = outFeatures;

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int i = 0;
        int limit4 = seqLen - (seqLen % 4);

        for (; i < limit4; i += 4)
        {
            s0 += gradOutput[i * outF + j];
            s1 += gradOutput[(i + 1) * outF + j];
            s2 += gradOutput[(i + 2) * outF + j];
            s3 += gradOutput[(i + 3) * outF + j];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; i < seqLen; i++)
            sum += gradOutput[i * outF + j];

        gradBias[j] = sum;
    }

    // ─────────────────────────────────────────────────────────────
    // InputGrad: gradInput[i,k] = sum_j(gradOutput[i,j] * weight[k,j])
    // Разворот x4 по outFeatures
    // weight[k,j] читается как weight[k*outF + j] — для разных k в warp
    // ─────────────────────────────────────────────────────────────
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

        int inF = inFeatures;
        int outF = outFeatures;

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int j = 0;
        int limit4 = outF - (outF % 4);

        for (; j < limit4; j += 4)
        {
            // gradOutput[i, j..j+3]: последовательный доступ
            // weight[k, j..j+3]: последовательный доступ
            s0 += gradOutput[i * outF + j] * weight[k * outF + j];
            s1 += gradOutput[i * outF + j + 1] * weight[k * outF + j + 1];
            s2 += gradOutput[i * outF + j + 2] * weight[k * outF + j + 2];
            s3 += gradOutput[i * outF + j + 3] * weight[k * outF + j + 3];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; j < outF; j++)
            sum += gradOutput[i * outF + j] * weight[k * outF + j];

        gradInput[i * inF + k] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════════
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

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _forwardKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _weightGradKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _biasGradKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
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

        _forwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            LinearKernels.LinearForwardKernel);

        _weightGradKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            LinearKernels.LinearWeightGradKernel);

        _biasGradKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            LinearKernels.LinearBiasGradKernel);

        _inputGradKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
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
            throw new ArgumentException(
                $"seqLen должен быть > 0, получено {seqLen}");

        int inputSize = seqLen * _inFeatures;
        int outputSize = seqLen * _outFeatures;
        _cachedSeqLen = seqLen;
        _cachedInputView = input.SubView(0, inputSize);

        var outputView = _outputBuffer.View.SubView(0, outputSize);

        _forwardKernel(new Index2D(seqLen, _outFeatures),
            _cachedInputView,
            _weight.View,
            _bias.View,
            outputView,
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
        int inputSize = seqLen * _inFeatures;

        // WeightGrad и BiasGrad можно запустить независимо (нет зависимостей)
        _weightGradKernel(new Index2D(_inFeatures, _outFeatures),
            _cachedInputView,
            grads,
            _gradWeight.View,
            seqLen,
            SpecializedValue.New(_inFeatures),
            SpecializedValue.New(_outFeatures));

        _biasGradKernel(new Index1D(_outFeatures),
            grads,
            _gradBias.View,
            seqLen,
            SpecializedValue.New(_outFeatures));

        var gradInputView = _gradInputBuffer.View.SubView(0, inputSize);

        _inputGradKernel(new Index2D(seqLen, _inFeatures),
            grads,
            _weight.View,
            gradInputView,
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