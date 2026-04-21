//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class MultiHeadAttentionKernels
{
    // ─────────────────────────────────────────────────────────────
    // Scores + CausalMask — без изменений, уже оптимизирован
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadScoresKernel(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> q,
        ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> scores,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim,
        SpecializedValue<float> scale)
    {
        int totalElements = numHeads * seqLen * seqLen;
        if (linearIdx >= totalElements) return;

        int h = linearIdx / (seqLen * seqLen);
        int remainder = linearIdx % (seqLen * seqLen);
        int i = remainder / seqLen;
        int j = remainder % seqLen;

        if (j > i)
        {
            scores[linearIdx] = -3.4028235e+38f;
            return;
        }

        int headDimVal = headDim;
        int qBase = i * numHeads * headDimVal + h * headDimVal;
        int kBase = j * numHeads * headDimVal + h * headDimVal;

        float dot0 = 0f, dot1 = 0f, dot2 = 0f, dot3 = 0f;
        int d = 0;
        int unroll4 = headDimVal & ~3;

        for (; d < unroll4; d += 4)
        {
            dot0 += q[qBase + d] * k[kBase + d];
            dot1 += q[qBase + d + 1] * k[kBase + d + 1];
            dot2 += q[qBase + d + 2] * k[kBase + d + 2];
            dot3 += q[qBase + d + 3] * k[kBase + d + 3];
        }
        float dot = dot0 + dot1 + dot2 + dot3;
        for (; d < headDimVal; d++)
            dot += q[qBase + d] * k[kBase + d];

        scores[linearIdx] = dot * scale;
    }

    // ─────────────────────────────────────────────────────────────
    // Softmax: объединяем проход max и exp в один с развёрткой x4
    // Алгоритм: проход 1 = max, проход 2 = exp+sum+normalize
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadSoftmaxKernel(
        Index1D headRow,
        ArrayView1D<float, Stride1D.Dense> scores,
        int seqLen,
        SpecializedValue<int> numHeads)
    {
        int totalRows = numHeads * seqLen;
        if (headRow >= totalRows) return;

        int h = headRow / seqLen;
        int i = headRow % seqLen;
        int offset = h * seqLen * seqLen + i * seqLen;
        int validLen = i + 1;

        // Проход 1: max с развёрткой x4
        float max0 = -3.4028235e+38f, max1 = -3.4028235e+38f;
        float max2 = -3.4028235e+38f, max3 = -3.4028235e+38f;
        int j = 0;
        int limit4 = validLen - (validLen % 4);

        for (; j < limit4; j += 4)
        {
            float v0 = scores[offset + j];
            float v1 = scores[offset + j + 1];
            float v2 = scores[offset + j + 2];
            float v3 = scores[offset + j + 3];
            if (v0 > max0) max0 = v0;
            if (v1 > max1) max1 = v1;
            if (v2 > max2) max2 = v2;
            if (v3 > max3) max3 = v3;
        }
        float maxVal = XMath.Max(XMath.Max(max0, max1), XMath.Max(max2, max3));
        for (; j < validLen; j++)
        {
            float v = scores[offset + j];
            if (v > maxVal) maxVal = v;
        }

        // Проход 2: exp + sum с развёрткой x4
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        j = 0;
        for (; j < limit4; j += 4)
        {
            float e0 = XMath.Exp(scores[offset + j] - maxVal);
            float e1 = XMath.Exp(scores[offset + j + 1] - maxVal);
            float e2 = XMath.Exp(scores[offset + j + 2] - maxVal);
            float e3 = XMath.Exp(scores[offset + j + 3] - maxVal);
            scores[offset + j] = e0;
            scores[offset + j + 1] = e1;
            scores[offset + j + 2] = e2;
            scores[offset + j + 3] = e3;
            s0 += e0; s1 += e1; s2 += e2; s3 += e3;
        }
        float sumExp = s0 + s1 + s2 + s3;
        for (; j < validLen; j++)
        {
            float e = XMath.Exp(scores[offset + j] - maxVal);
            scores[offset + j] = e;
            sumExp += e;
        }

        // Проход 3: нормировка с развёрткой x4
        float invSum = 1.0f / (sumExp + 1e-10f);
        j = 0;
        for (; j < limit4; j += 4)
        {
            scores[offset + j] *= invSum;
            scores[offset + j + 1] *= invSum;
            scores[offset + j + 2] *= invSum;
            scores[offset + j + 3] *= invSum;
        }
        for (; j < validLen; j++)
            scores[offset + j] *= invSum;
    }

    // ─────────────────────────────────────────────────────────────
    // WeightedSum: добавляем развёртку по j
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadWeightedSumKernel(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> attnWeights,
        ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> output,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim)
    {
        int totalElements = seqLen * numHeads * headDim;
        if (linearIdx >= totalElements) return;

        int i = linearIdx / (numHeads * headDim);
        int remainder = linearIdx % (numHeads * headDim);
        int h = remainder / headDim;
        int d = remainder % headDim;

        int attnBase = h * seqLen * seqLen + i * seqLen;
        int vStride = numHeads * headDim;
        int vBase = h * headDim + d;
        int validJ = i + 1;

        // Развёртка x4 по j
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int j = 0;
        int limit4 = validJ - (validJ % 4);

        for (; j < limit4; j += 4)
        {
            s0 += attnWeights[attnBase + j] * v[j * vStride + vBase];
            s1 += attnWeights[attnBase + j + 1] * v[(j + 1) * vStride + vBase];
            s2 += attnWeights[attnBase + j + 2] * v[(j + 2) * vStride + vBase];
            s3 += attnWeights[attnBase + j + 3] * v[(j + 3) * vStride + vBase];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; j < validJ; j++)
            sum += attnWeights[attnBase + j] * v[j * vStride + vBase];

        output[linearIdx] = sum;
    }

    // ─────────────────────────────────────────────────────────────
    // GradQ — без изменений, уже оптимизирован
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadGradQKernel_NoAtomic(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> gradScores,
        ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> gradQ,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim,
        SpecializedValue<float> invScale)
    {
        int total = seqLen * numHeads * headDim;
        if (linearIdx >= total) return;

        int i = linearIdx / (numHeads * headDim);
        int rem = linearIdx % (numHeads * headDim);
        int h = rem / headDim;
        int d = rem % headDim;
        int gsBase = h * seqLen * seqLen + i * seqLen;
        int kStride = numHeads * headDim;
        int kBase = h * headDim + d;

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int j = 0;
        int limit4 = (i + 1) - ((i + 1) % 4);

        for (; j < limit4; j += 4)
        {
            s0 += gradScores[gsBase + j] * k[j * kStride + kBase];
            s1 += gradScores[gsBase + j + 1] * k[(j + 1) * kStride + kBase];
            s2 += gradScores[gsBase + j + 2] * k[(j + 2) * kStride + kBase];
            s3 += gradScores[gsBase + j + 3] * k[(j + 3) * kStride + kBase];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; j <= i; j++)
            sum += gradScores[gsBase + j] * k[j * kStride + kBase];

        gradQ[linearIdx] = sum * invScale;
    }

    // ─────────────────────────────────────────────────────────────
    // GradK с развёрткой по i
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadGradKKernel_NoAtomic(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> gradScores,
        ArrayView1D<float, Stride1D.Dense> q,
        ArrayView1D<float, Stride1D.Dense> gradK,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim,
        SpecializedValue<float> invScale)
    {
        int total = seqLen * numHeads * headDim;
        if (linearIdx >= total) return;

        int j = linearIdx / (numHeads * headDim);
        int rem = linearIdx % (numHeads * headDim);
        int h = rem / headDim;
        int d = rem % headDim;
        int qStride = numHeads * headDim;
        int qBase = h * headDim + d;

        int remaining = seqLen - j;
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int i = j;
        int limit4 = j + (remaining - (remaining % 4));

        for (; i < limit4; i += 4)
        {
            s0 += gradScores[h * seqLen * seqLen + i * seqLen + j]
                  * q[i * qStride + qBase];
            s1 += gradScores[h * seqLen * seqLen + (i + 1) * seqLen + j]
                  * q[(i + 1) * qStride + qBase];
            s2 += gradScores[h * seqLen * seqLen + (i + 2) * seqLen + j]
                  * q[(i + 2) * qStride + qBase];
            s3 += gradScores[h * seqLen * seqLen + (i + 3) * seqLen + j]
                  * q[(i + 3) * qStride + qBase];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; i < seqLen; i++)
            sum += gradScores[h * seqLen * seqLen + i * seqLen + j]
                   * q[i * qStride + qBase];

        gradK[linearIdx] = sum * invScale;
    }

    // ─────────────────────────────────────────────────────────────
    // AccumThree: dst = a + b + c за один проход (вместо двух Add)
    // ─────────────────────────────────────────────────────────────
    public static void AccumThreeKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> dst,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> c)
    {
        dst[idx] = a[idx] + b[idx] + c[idx];
    }

    public static void AccumAddKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> dst,
        ArrayView1D<float, Stride1D.Dense> src)
    {
        dst[idx] += src[idx];
    }

    // ─────────────────────────────────────────────────────────────
    // GradV — с развёрткой по i
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadGradVKernel(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> attnWeights,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradV,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim)
    {
        int totalElements = seqLen * numHeads * headDim;
        if (linearIdx >= totalElements) return;

        int j = linearIdx / (numHeads * headDim);
        int remainder = linearIdx % (numHeads * headDim);
        int h = remainder / headDim;
        int d = remainder % headDim;

        int attnBase = h * seqLen * seqLen;
        int goBase = h * headDim + d;
        int stride = numHeads * headDim;

        int remaining = seqLen - j;
        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int i = j;
        int limit4 = j + (remaining - (remaining % 4));

        for (; i < limit4; i += 4)
        {
            s0 += attnWeights[attnBase + i * seqLen + j]
                  * gradOutput[i * stride + goBase];
            s1 += attnWeights[attnBase + (i + 1) * seqLen + j]
                  * gradOutput[(i + 1) * stride + goBase];
            s2 += attnWeights[attnBase + (i + 2) * seqLen + j]
                  * gradOutput[(i + 2) * stride + goBase];
            s3 += attnWeights[attnBase + (i + 3) * seqLen + j]
                  * gradOutput[(i + 3) * stride + goBase];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; i < seqLen; i++)
            sum += attnWeights[attnBase + i * seqLen + j]
                   * gradOutput[i * stride + goBase];

        gradV[linearIdx] = sum;
    }

    // ─────────────────────────────────────────────────────────────
    // GradAttnWeights — без изменений (уже развёртка x4)
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadGradAttnWeightsKernel(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> v,
        ArrayView1D<float, Stride1D.Dense> gradAttnWeights,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim)
    {
        int totalElements = numHeads * seqLen * seqLen;
        if (linearIdx >= totalElements) return;

        int h = linearIdx / (seqLen * seqLen);
        int remainder = linearIdx % (seqLen * seqLen);
        int i = remainder / seqLen;
        int j = remainder % seqLen;

        if (j > i) { gradAttnWeights[linearIdx] = 0; return; }

        int goBase = i * numHeads * headDim + h * headDim;
        int vBase = j * numHeads * headDim + h * headDim;
        int hd = headDim;

        float s0 = 0f, s1 = 0f, s2 = 0f, s3 = 0f;
        int d = 0;
        int u4 = hd & ~3;
        for (; d < u4; d += 4)
        {
            s0 += gradOutput[goBase + d] * v[vBase + d];
            s1 += gradOutput[goBase + d + 1] * v[vBase + d + 1];
            s2 += gradOutput[goBase + d + 2] * v[vBase + d + 2];
            s3 += gradOutput[goBase + d + 3] * v[vBase + d + 3];
        }
        float sum = s0 + s1 + s2 + s3;
        for (; d < hd; d++)
            sum += gradOutput[goBase + d] * v[vBase + d];

        gradAttnWeights[linearIdx] = sum;
    }

    // ─────────────────────────────────────────────────────────────
    // SoftmaxBackward: развёртка x4 для dotProduct и gradScores
    // ─────────────────────────────────────────────────────────────
    public static void MultiHeadSoftmaxBackwardKernel(
        Index1D headRow,
        ArrayView1D<float, Stride1D.Dense> attnWeights,
        ArrayView1D<float, Stride1D.Dense> gradAttnWeights,
        ArrayView1D<float, Stride1D.Dense> gradScores,
        int seqLen,
        SpecializedValue<int> numHeads)
    {
        int totalRows = numHeads * seqLen;
        if (headRow >= totalRows) return;

        int h = headRow / seqLen;
        int i = headRow % seqLen;
        int offset = h * seqLen * seqLen + i * seqLen;
        int validLen = i + 1;
        int limit4 = validLen - (validLen % 4);

        // Проход 1: dotProduct с развёрткой x4
        float dp0 = 0f, dp1 = 0f, dp2 = 0f, dp3 = 0f;
        int j = 0;
        for (; j < limit4; j += 4)
        {
            dp0 += attnWeights[offset + j] * gradAttnWeights[offset + j];
            dp1 += attnWeights[offset + j + 1] * gradAttnWeights[offset + j + 1];
            dp2 += attnWeights[offset + j + 2] * gradAttnWeights[offset + j + 2];
            dp3 += attnWeights[offset + j + 3] * gradAttnWeights[offset + j + 3];
        }
        float dotProduct = dp0 + dp1 + dp2 + dp3;
        for (; j < validLen; j++)
            dotProduct += attnWeights[offset + j] * gradAttnWeights[offset + j];

        // Проход 2: gradScores с развёрткой x4
        j = 0;
        for (; j < limit4; j += 4)
        {
            float w0 = attnWeights[offset + j];
            float w1 = attnWeights[offset + j + 1];
            float w2 = attnWeights[offset + j + 2];
            float w3 = attnWeights[offset + j + 3];
            gradScores[offset + j] = w0 * (gradAttnWeights[offset + j] - dotProduct);
            gradScores[offset + j + 1] = w1 * (gradAttnWeights[offset + j + 1] - dotProduct);
            gradScores[offset + j + 2] = w2 * (gradAttnWeights[offset + j + 2] - dotProduct);
            gradScores[offset + j + 3] = w3 * (gradAttnWeights[offset + j + 3] - dotProduct);
        }
        for (; j < validLen; j++)
            gradScores[offset + j] =
                attnWeights[offset + j] * (gradAttnWeights[offset + j] - dotProduct);

        // Маскированные позиции
        for (j = validLen; j < seqLen; j++)
            gradScores[offset + j] = 0f;
    }
}

// ═══════════════════════════════════════════════════════════════════
public class MultiHeadAttentionLayer : ILayer
{
    private readonly Accelerator _accelerator;
    public readonly int EmbeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;

    // ✅ invScale вычисляется один раз в конструкторе
    private readonly float _invScale;

    internal LinearLayer _wq, _wk, _wv, _wo;

    private MemoryBuffer1D<float, Stride1D.Dense> _inputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _qBuffer, _kBuffer, _vBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _scoresBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _attnOutputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradAttnWeightsBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradScoresBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradQBuffer, _gradKBuffer, _gradVBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradInputBuffer;

    private ArrayView1D<float, Stride1D.Dense> _cachedInputView;
    private ArrayView1D<float, Stride1D.Dense> _cachedQView, _cachedKView, _cachedVView;
    private ArrayView1D<float, Stride1D.Dense> _cachedAttnWeightsView;
    private int _cachedSeqLen;

    // ── Kernels ──
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>,
        SpecializedValue<float>> _scoresKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _softmaxKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _weightedSumKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _gradVKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _gradAttnWeightsKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _softmaxBackwardKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>,
        SpecializedValue<float>> _gradQKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>,
        SpecializedValue<float>> _gradKKernel;

    // ✅ AccumThree вместо двух AccumAdd
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _accumThreeKernel;

    private bool _disposed;

    public MultiHeadAttentionLayer(
        Accelerator accelerator,
        int embeddingDim,
        int numHeads = 4,
        int maxSeqLen = 80)
    {
        _accelerator = accelerator;
        EmbeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _maxSeqLen = maxSeqLen;

        if (embeddingDim % numHeads != 0)
            throw new ArgumentException(
                $"embeddingDim ({embeddingDim}) " +
                $"должен делиться на numHeads ({numHeads})");

        // ✅ Вычисляем один раз
        _invScale = 1.0f / MathF.Sqrt(_headDim);

        _wq = new LinearLayer(accelerator, embeddingDim, embeddingDim, false, maxSeqLen);
        _wk = new LinearLayer(accelerator, embeddingDim, embeddingDim, false, maxSeqLen);
        _wv = new LinearLayer(accelerator, embeddingDim, embeddingDim, false, maxSeqLen);
        _wo = new LinearLayer(accelerator, embeddingDim, embeddingDim, true, maxSeqLen);

        int maxInput = maxSeqLen * embeddingDim;
        int maxScores = numHeads * maxSeqLen * maxSeqLen;

        _inputBuffer = accelerator.Allocate1D<float>(maxInput);
        _qBuffer = accelerator.Allocate1D<float>(maxInput);
        _kBuffer = accelerator.Allocate1D<float>(maxInput);
        _vBuffer = accelerator.Allocate1D<float>(maxInput);
        _scoresBuffer = accelerator.Allocate1D<float>(maxScores);
        _attnOutputBuffer = accelerator.Allocate1D<float>(maxInput);
        _gradAttnWeightsBuffer = accelerator.Allocate1D<float>(maxScores);
        _gradScoresBuffer = accelerator.Allocate1D<float>(maxScores);
        _gradQBuffer = accelerator.Allocate1D<float>(maxInput);
        _gradKBuffer = accelerator.Allocate1D<float>(maxInput);
        _gradVBuffer = accelerator.Allocate1D<float>(maxInput);
        _gradInputBuffer = accelerator.Allocate1D<float>(maxInput);

        _scoresKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>,
            SpecializedValue<float>>(
            MultiHeadAttentionKernels.MultiHeadScoresKernel);

        _softmaxKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadSoftmaxKernel);

        _weightedSumKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadWeightedSumKernel);

        _gradVKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadGradVKernel);

        _gradAttnWeightsKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadGradAttnWeightsKernel);

        _softmaxBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadSoftmaxBackwardKernel);

        _gradQKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>,
            SpecializedValue<float>>(
            MultiHeadAttentionKernels.MultiHeadGradQKernel_NoAtomic);

        _gradKKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>,
            SpecializedValue<float>>(
            MultiHeadAttentionKernels.MultiHeadGradKKernel_NoAtomic);

        // ✅ AccumThree — один kernel вместо двух AccumAdd
        _accumThreeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            MultiHeadAttentionKernels.AccumThreeKernel);
    }

    public string LayerType => "MultiHeadAttention";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        int inputSize = seqLen * EmbeddingDim;
        int scoresSize = _numHeads * seqLen * seqLen;
        _cachedSeqLen = seqLen;

        var inputView = _inputBuffer.View.SubView(0, inputSize);
        inputView.CopyFrom(input.SubView(0, inputSize));
        _cachedInputView = inputView;

        var Q = _wq.Forward(_cachedInputView, seqLen);
        var K = _wk.Forward(_cachedInputView, seqLen);
        var V = _wv.Forward(_cachedInputView, seqLen);

        _qBuffer.View.SubView(0, inputSize).CopyFrom(Q);
        _kBuffer.View.SubView(0, inputSize).CopyFrom(K);
        _vBuffer.View.SubView(0, inputSize).CopyFrom(V);

        _cachedQView = _qBuffer.View.SubView(0, inputSize);
        _cachedKView = _kBuffer.View.SubView(0, inputSize);
        _cachedVView = _vBuffer.View.SubView(0, inputSize);

        var scoresView = _scoresBuffer.View.SubView(0, scoresSize);

        _scoresKernel(
            scoresSize,
            _cachedQView, _cachedKView, scoresView,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim),
            SpecializedValue.New(_invScale)); // ✅ из поля

        _softmaxKernel(
            _numHeads * seqLen,
            scoresView, seqLen,
            SpecializedValue.New(_numHeads));

        _cachedAttnWeightsView = scoresView;

        var attnOut = _attnOutputBuffer.View.SubView(0, inputSize);
        _weightedSumKernel(
            inputSize,
            scoresView, _cachedVView, attnOut,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim));

        return _wo.Forward(attnOut, seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr)
    {
        if (_cachedSeqLen == 0)
            throw new InvalidOperationException(
                "Forward must be called before Backward");

        int seqLen = _cachedSeqLen;
        int inputSize = seqLen * EmbeddingDim;
        int scoresSize = _numHeads * seqLen * seqLen;

        var gradAttnOut = _wo.Backward(grads, lr);

        var gradVView = _gradVBuffer.View.SubView(0, inputSize);
        _gradVKernel(
            inputSize,
            _cachedAttnWeightsView, gradAttnOut, gradVView,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim));

        var gradAttnWView = _gradAttnWeightsBuffer.View.SubView(0, scoresSize);
        _gradAttnWeightsKernel(
            scoresSize,
            gradAttnOut, _cachedVView, gradAttnWView,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim));

        var gradScoresView = _gradScoresBuffer.View.SubView(0, scoresSize);
        _softmaxBackwardKernel(
            _numHeads * seqLen,
            _cachedAttnWeightsView, gradAttnWView, gradScoresView,
            seqLen,
            SpecializedValue.New(_numHeads));

        var gradQView = _gradQBuffer.View.SubView(0, inputSize);
        var gradKView = _gradKBuffer.View.SubView(0, inputSize);

        // Независимые kernels — GPU может перекрывать их выполнение
        _gradQKernel(
            inputSize,
            gradScoresView, _cachedKView, gradQView,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim),
            SpecializedValue.New(_invScale)); // ✅ из поля

        _gradKKernel(
            inputSize,
            gradScoresView, _cachedQView, gradKView,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim),
            SpecializedValue.New(_invScale)); // ✅ из поля

        var gradInputQ = _wq.Backward(gradQView, lr);
        var gradInputK = _wk.Backward(gradKView, lr);
        var gradInputV = _wv.Backward(gradVView, lr);

        // ✅ AccumThree: один kernel вместо двух AccumAdd
        // 3 чтения + 1 запись вместо 2 чтений + 2 записей
        var gradInputView = _gradInputBuffer.View.SubView(0, inputSize);
        _accumThreeKernel(
            inputSize,
            gradInputView,
            gradInputQ.SubView(0, inputSize),
            gradInputK.SubView(0, inputSize),
            gradInputV.SubView(0, inputSize));

        return gradInputView;
    }

    public int Parameters() =>
        _wq.Parameters() + _wk.Parameters() +
        _wv.Parameters() + _wo.Parameters();

    public void Dispose()
    {
        if (!_disposed)
        {
            _wq.Dispose(); _wk.Dispose(); _wv.Dispose(); _wo.Dispose();
            _inputBuffer.Dispose();
            _qBuffer.Dispose(); _kBuffer.Dispose(); _vBuffer.Dispose();
            _scoresBuffer.Dispose(); _attnOutputBuffer.Dispose();
            _gradAttnWeightsBuffer.Dispose(); _gradScoresBuffer.Dispose();
            _gradQBuffer.Dispose(); _gradKBuffer.Dispose();
            _gradVBuffer.Dispose(); _gradInputBuffer.Dispose();
            _disposed = true;
        }
    }
}