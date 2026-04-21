//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

// LossKernels без изменений — уже хорошо оптимизированы
public static class LossKernels
{
    public static void FindMaxKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> logits,
        ArrayView1D<float, Stride1D.Dense> maxVals,
        SpecializedValue<int> vocabSize)
    {
        int vSize = vocabSize;
        int offset = row * vSize;
        float max0 = float.NegativeInfinity, max1 = float.NegativeInfinity;
        float max2 = float.NegativeInfinity, max3 = float.NegativeInfinity;
        int i = 0;
        int limit4 = vSize - (vSize % 4);
        for (; i < limit4; i += 4)
        {
            float v0 = logits[offset + i];
            float v1 = logits[offset + i + 1];
            float v2 = logits[offset + i + 2];
            float v3 = logits[offset + i + 3];
            if (v0 > max0) max0 = v0;
            if (v1 > max1) max1 = v1;
            if (v2 > max2) max2 = v2;
            if (v3 > max3) max3 = v3;
        }
        float maxVal = XMath.Max(XMath.Max(max0, max1), XMath.Max(max2, max3));
        for (; i < vSize; i++) { float v = logits[offset + i]; if (v > maxVal) maxVal = v; }
        maxVals[row] = maxVal;
    }

    public static void SoftmaxAndLossKernel(
        Index1D row,
        ArrayView1D<float, Stride1D.Dense> logits,
        ArrayView1D<float, Stride1D.Dense> maxVals,
        ArrayView1D<int, Stride1D.Dense> targetIds,
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<float, Stride1D.Dense> lossBuffer,
        SpecializedValue<int> vocabSize)
    {
        int vSize = vocabSize;
        int offset = row * vSize;
        float maxVal = maxVals[row];
        int targetIdx = targetIds[row];

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int i = 0;
        int limit4 = vSize - (vSize % 4);
        for (; i < limit4; i += 4)
        {
            float e0 = XMath.Exp(logits[offset + i] - maxVal);
            float e1 = XMath.Exp(logits[offset + i + 1] - maxVal);
            float e2 = XMath.Exp(logits[offset + i + 2] - maxVal);
            float e3 = XMath.Exp(logits[offset + i + 3] - maxVal);
            probs[offset + i] = e0; probs[offset + i + 1] = e1;
            probs[offset + i + 2] = e2; probs[offset + i + 3] = e3;
            s0 += e0; s1 += e1; s2 += e2; s3 += e3;
        }
        float sumExp = s0 + s1 + s2 + s3;
        for (; i < vSize; i++)
        {
            float e = XMath.Exp(logits[offset + i] - maxVal);
            probs[offset + i] = e; sumExp += e;
        }

        float invSum = 1.0f / sumExp;
        float targetProb = 0f;
        i = 0;
        for (; i < limit4; i += 4)
        {
            float p0 = probs[offset + i] * invSum;
            float p1 = probs[offset + i + 1] * invSum;
            float p2 = probs[offset + i + 2] * invSum;
            float p3 = probs[offset + i + 3] * invSum;
            probs[offset + i] = p0; probs[offset + i + 1] = p1;
            probs[offset + i + 2] = p2; probs[offset + i + 3] = p3;
            if (i == targetIdx) targetProb = p0;
            if (i + 1 == targetIdx) targetProb = p1;
            if (i + 2 == targetIdx) targetProb = p2;
            if (i + 3 == targetIdx) targetProb = p3;
        }
        for (; i < vSize; i++)
        {
            float p = probs[offset + i] * invSum;
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
        SpecializedValue<int> vocabSize)
    {
        int vSize = vocabSize;
        int row = index / vSize;
        int col = index % vSize;
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
        ArrayView1D<float, Stride1D.Dense> totalLossBuffer,
        SpecializedValue<int> seqLen)
    {
        int i = index * 4;
        int total = seqLen;
        float localSum = 0f;
        if (i + 3 < total)
        {
            localSum = lossBuffer[i] + lossBuffer[i + 1]
                     + lossBuffer[i + 2] + lossBuffer[i + 3];
        }
        else
        {
            for (int j = i; j < total; j++) localSum += lossBuffer[j];
        }
        if (localSum != 0f)
            Atomic.Add(ref totalLossBuffer[0], localSum);
    }
}

public class LossManager : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _vocabSize;
    private readonly int _maxSeqLen;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _targetIdsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _maxValsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _lossBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _probsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _gradsBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _totalLossBuffer;

    // ── ОПТИМИЗАЦИЯ: пинированный буфер вместо new float[1] ──────
    // Оригинал: new float[1] на каждый TrainStep = GC давление
    // Новый: один пинированный буфер на весь lifetime LossManager
    private readonly float[] _hostLossBuf;

    private int _cachedSeqLen;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _findMaxKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _softmaxLossKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _gradsKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        float> _normalizeGradsKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _sumLossKernel;

    private bool _disposed;

    public LossManager(Accelerator accelerator, int vocabSize, int maxSeqLen)
    {
        _accelerator = accelerator;
        _vocabSize = vocabSize;
        _maxSeqLen = maxSeqLen;

        _targetIdsBuffer = accelerator.Allocate1D<int>(maxSeqLen);
        _maxValsBuffer = accelerator.Allocate1D<float>(maxSeqLen);
        _lossBuffer = accelerator.Allocate1D<float>(maxSeqLen);
        _probsBuffer = accelerator.Allocate1D<float>(maxSeqLen * vocabSize);
        _gradsBuffer = accelerator.Allocate1D<float>(maxSeqLen * vocabSize);
        _totalLossBuffer = accelerator.Allocate1D<float>(1);

        // Пинированный буфер — один раз на весь lifetime
        _hostLossBuf = GC.AllocateArray<float>(1, pinned: true);

        _findMaxKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(LossKernels.FindMaxKernel);

        _softmaxLossKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(LossKernels.SoftmaxAndLossKernel);

        _gradsKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(LossKernels.ComputeGradientsKernel);

        _normalizeGradsKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float>(LossKernels.NormalizeGradientsKernel);

        _sumLossKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(LossKernels.SumLossKernel);
    }

    public (ArrayView1D<float, Stride1D.Dense> probs, float loss)
        ComputeSoftmaxAndLoss(
            ArrayView1D<float, Stride1D.Dense> logits,
            ArrayView1D<int, Stride1D.Dense> targetIdsView,
            int seqLen)
    {
        int totalSize = seqLen * _vocabSize;

        _targetIdsBuffer.View.SubView(0, seqLen).CopyFrom(targetIdsView);
        _cachedSeqLen = seqLen;

        _findMaxKernel(
            seqLen, logits,
            _maxValsBuffer.View.SubView(0, seqLen),
            SpecializedValue.New(_vocabSize));

        _softmaxLossKernel(
            seqLen, logits,
            _maxValsBuffer.View.SubView(0, seqLen),
            _targetIdsBuffer.View.SubView(0, seqLen),
            _probsBuffer.View.SubView(0, totalSize),
            _lossBuffer.View.SubView(0, seqLen),
            SpecializedValue.New(_vocabSize));

        _totalLossBuffer.MemSetToZero();

        int sumThreads = (seqLen + 3) / 4;
        _sumLossKernel(
            sumThreads,
            _lossBuffer.View.SubView(0, seqLen),
            _totalLossBuffer.View,
            SpecializedValue.New(seqLen));

        // ── Используем пинированный буфер — нет GC на каждый шаг ──
        _totalLossBuffer.CopyToCPU(_hostLossBuf);

        return (_probsBuffer.View.SubView(0, totalSize),
                _hostLossBuf[0] / seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> ComputeGradients(
        ArrayView1D<float, Stride1D.Dense> probs,
        ArrayView1D<int, Stride1D.Dense> targetIdsView,
        int seqLen)
    {
        if (seqLen != _cachedSeqLen)
        {
            _targetIdsBuffer.View.SubView(0, seqLen).CopyFrom(targetIdsView);
            _cachedSeqLen = seqLen;
        }

        int totalSize = seqLen * _vocabSize;
        float invSeqLen = 1.0f / seqLen;

        _gradsKernel(
            totalSize,
            probs,
            _targetIdsBuffer.View.SubView(0, seqLen),
            _gradsBuffer.View.SubView(0, totalSize),
            SpecializedValue.New(_vocabSize));

        _normalizeGradsKernel(
            totalSize,
            _gradsBuffer.View.SubView(0, totalSize),
            invSeqLen);

        return _gradsBuffer.View.SubView(0, totalSize);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _targetIdsBuffer.Dispose();
            _maxValsBuffer.Dispose();
            _lossBuffer.Dispose();
            _probsBuffer.Dispose();
            _gradsBuffer.Dispose();
            _totalLossBuffer.Dispose();
            _disposed = true;
        }
    }
}