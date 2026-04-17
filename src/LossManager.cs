//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class LossKernels
{
    public static void FindMaxKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> logits,
        ArrayView1D<float, Stride1D.Dense> maxVals,
        int vocabSize)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            float val = logits[row * vocabSize + i];
            if (val > max) max = val;
        }
        maxVals[row] = max;
    }

    public static void SoftmaxAndLossKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> logits,
        ArrayView1D<float, Stride1D.Dense> maxVals,
        ArrayView1D<int, Stride1D.Dense> targetIds,
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<float, Stride1D.Dense> lossBuffer,
        int vocabSize)
    {
        int offset = row * vocabSize;
        float maxVal = maxVals[row];
        int targetIdx = targetIds[row];

        float sumExp = 0;
        for (int i = 0; i < vocabSize; i++)
            sumExp += XMath.Exp(logits[offset + i] - maxVal);

        float targetProb = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            float p = XMath.Exp(logits[offset + i] - maxVal) / sumExp;
            probs[offset + i] = p;
            if (i == targetIdx) targetProb = p;
        }

        lossBuffer[row] = -XMath.Log(XMath.Max(targetProb, 1e-10f));
    }

    public static void ComputeGradientsKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<int, Stride1D.Dense> targetIds,
        ArrayView1D<float, Stride1D.Dense> grads,
        int vocabSize)
    {
        int row = index / vocabSize;
        int col = index % vocabSize;
        float grad = probs[index];
        if (col == targetIds[row]) grad -= 1.0f;
        grads[index] = grad;
    }

    public static void NormalizeGradientsKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> grads,
        float invSeqLen)
    {
        grads[index] *= invSeqLen;
    }

    public static void SumLossKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> lossBuffer,
        ArrayView1D<float, Stride1D.Dense> totalLossBuffer)
    {
        Atomic.Add(ref totalLossBuffer[0], lossBuffer[index]);
    }
}

public class LossManager : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _vocabSize;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _targetIdsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _maxValsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _lossBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _probsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _gradsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _totalLossBuffer;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int> _findMaxKernel;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int> _softmaxLossKernel;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int> _gradsKernel;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        float> _normalizeGradsKernel;
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _sumLossKernel;

    private bool _disposed;

    public LossManager(Accelerator accelerator, int vocabSize, int maxSeqLen)
    {
        _accelerator = accelerator;
        _vocabSize = vocabSize;

        _targetIdsBuffer = accelerator.Allocate1D<int>(maxSeqLen);
        _maxValsBuffer = accelerator.Allocate1D<float>(maxSeqLen);
        _lossBuffer = accelerator.Allocate1D<float>(maxSeqLen);
        _probsBuffer = accelerator.Allocate1D<float>(maxSeqLen * vocabSize);
        _gradsBuffer = accelerator.Allocate1D<float>(maxSeqLen * vocabSize);
        _totalLossBuffer = accelerator.Allocate1D<float>(1);

        _findMaxKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            int>(LossKernels.FindMaxKernel);

        _softmaxLossKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(
            LossKernels.SoftmaxAndLossKernel);

        _gradsKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(
            LossKernels.ComputeGradientsKernel);

        _normalizeGradsKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, float>(
            LossKernels.NormalizeGradientsKernel);

        _sumLossKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(LossKernels.SumLossKernel);
    }

    public (ArrayView1D<float, Stride1D.Dense> probs, float loss)
        ComputeSoftmaxAndLoss(
            ArrayView1D<float, Stride1D.Dense> logits,
            ArrayView1D<int, Stride1D.Dense> targetIdsView,
            int seqLen)
    {
        _targetIdsBuffer.View.SubView(0, seqLen).CopyFrom(targetIdsView);
        _findMaxKernel(seqLen, logits, _maxValsBuffer.View, _vocabSize);

        _softmaxLossKernel(seqLen, logits, _maxValsBuffer.View,
            _targetIdsBuffer.View.SubView(0, seqLen),
            _probsBuffer.View.SubView(0, seqLen * _vocabSize),
            _lossBuffer.View.SubView(0, seqLen), _vocabSize);

        _totalLossBuffer.MemSetToZero();
        _sumLossKernel(seqLen,
            _lossBuffer.View.SubView(0, seqLen), _totalLossBuffer.View);

        float[] hostLoss = new float[1];
        _totalLossBuffer.CopyToCPU(hostLoss);

        return (_probsBuffer.View.SubView(0, seqLen * _vocabSize),
                hostLoss[0] / seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> ComputeGradients(
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<int, Stride1D.Dense> targetIdsView,
        int seqLen)
    {
        _targetIdsBuffer.View.SubView(0, seqLen).CopyFrom(targetIdsView);

        int totalSize = seqLen * _vocabSize;
        _gradsKernel(totalSize, probs,
            _targetIdsBuffer.View.SubView(0, seqLen),
            _gradsBuffer.View.SubView(0, totalSize), _vocabSize);

        _normalizeGradsKernel(totalSize,
            _gradsBuffer.View.SubView(0, totalSize), 1.0f / seqLen);

        return _gradsBuffer.View.SubView(0, totalSize);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _targetIdsBuffer.Dispose(); _maxValsBuffer.Dispose();
            _lossBuffer.Dispose(); _probsBuffer.Dispose();
            _gradsBuffer.Dispose(); _totalLossBuffer.Dispose();
            _disposed = true;
        }
    }
}