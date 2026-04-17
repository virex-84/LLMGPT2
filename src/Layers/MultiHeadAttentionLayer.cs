//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class MultiHeadAttentionKernels
{
    /// <summary>
    /// Attention scores для каждой головы отдельно.
    /// Q,K layout: [seqLen, numHeads, headDim]
    /// scores layout: [numHeads, seqLen, seqLen]
    /// </summary>
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

        // Causal mask
        if (j > i)
        {
            scores[linearIdx] = -3.4028235e+38f;
            return;
        }

        // Q[i, h, :] · K[j, h, :]
        float dot = 0;
        int qOffset = i * numHeads * headDim + h * headDim;
        int kOffset = j * numHeads * headDim + h * headDim;
        for (int d = 0; d < headDim; d++)
            dot += q[qOffset + d] * k[kOffset + d];

        scores[linearIdx] = dot / scale;
    }

    /// <summary>
    /// Softmax по каждой строке каждой головы.
    /// </summary>
    public static void MultiHeadSoftmaxKernel(
        Index1D headRow,
        ArrayView1D<float, Stride1D.Dense> scores,
        int seqLen,
        SpecializedValue<int> numHeads)
    {
        int totalRows = numHeads * seqLen;
        if (headRow >= totalRows) return;

        int offset = headRow * seqLen;

        float maxVal = -3.4028235e+38f;
        for (int j = 0; j < seqLen; j++)
        {
            float val = scores[offset + j];
            if (val > maxVal) maxVal = val;
        }

        float sumExp = 0;
        for (int j = 0; j < seqLen; j++)
        {
            float val = XMath.Exp(scores[offset + j] - maxVal);
            scores[offset + j] = val;
            sumExp += val;
        }

        float invSum = 1.0f / (sumExp + 1e-10f);
        for (int j = 0; j < seqLen; j++)
            scores[offset + j] *= invSum;
    }

    /// <summary>
    /// Weighted sum: output[i,h,:] = Σ_j attn[h,i,j] * V[j,h,:]
    /// </summary>
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

        float sum = 0;
        int attnOffset = h * seqLen * seqLen + i * seqLen;
        for (int j = 0; j < seqLen; j++)
        {
            float w = attnWeights[attnOffset + j];
            sum += w * v[j * numHeads * headDim + h * headDim + d];
        }

        output[linearIdx] = sum;
    }

    // ═══════════════════════════════════════
    // BACKWARD KERNELS
    // ═══════════════════════════════════════

    /// <summary>
    /// gradV[j,h,d] = Σ_i attn[h,i,j] * gradOutput[i,h,d]
    /// </summary>
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

        float sum = 0;
        int attnOffset = h * seqLen * seqLen;
        for (int i = 0; i < seqLen; i++)
            sum += attnWeights[attnOffset + i * seqLen + j] *
                   gradOutput[i * numHeads * headDim + h * headDim + d];

        gradV[linearIdx] = sum;
    }

    /// <summary>
    /// gradAttn[h,i,j] = Σ_d gradOutput[i,h,d] * V[j,h,d]
    /// </summary>
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

        float sum = 0;
        for (int d = 0; d < headDim; d++)
            sum += gradOutput[i * numHeads * headDim + h * headDim + d] *
                   v[j * numHeads * headDim + h * headDim + d];

        gradAttnWeights[linearIdx] = sum;
    }

    /// <summary>
    /// Softmax backward для каждой головы.
    /// </summary>
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

        int offset = headRow * seqLen;

        float dotProduct = 0;
        for (int j = 0; j < seqLen; j++)
            dotProduct += attnWeights[offset + j] * gradAttnWeights[offset + j];

        for (int j = 0; j < seqLen; j++)
        {
            float w = attnWeights[offset + j];
            gradScores[offset + j] = w * (gradAttnWeights[offset + j] - dotProduct);
        }
    }

    /// <summary>
    /// gradQ[i,h,d] += Σ_j gradScores[h,i,j] * K[j,h,d] / scale
    /// gradK[j,h,d] += Σ_i gradScores[h,i,j] * Q[i,h,d] / scale
    /// Один поток на (h, i, j) пару.
    /// </summary>
    public static void MultiHeadGradQKKernel(
        Index1D linearIdx,
        ArrayView1D<float, Stride1D.Dense> gradScores,
        ArrayView1D<float, Stride1D.Dense> q,
        ArrayView1D<float, Stride1D.Dense> k,
        ArrayView1D<float, Stride1D.Dense> gradQ,
        ArrayView1D<float, Stride1D.Dense> gradK,
        int seqLen,
        SpecializedValue<int> numHeads,
        SpecializedValue<int> headDim,
        SpecializedValue<float> scale)
    {
        int totalPairs = numHeads * seqLen * seqLen;
        if (linearIdx >= totalPairs) return;

        int h = linearIdx / (seqLen * seqLen);
        int remainder = linearIdx % (seqLen * seqLen);
        int i = remainder / seqLen;
        int j = remainder % seqLen;

        if (j > i) return; // Causal mask

        float gs = gradScores[linearIdx] / scale;

        int qOffset = i * numHeads * headDim + h * headDim;
        int kOffset = j * numHeads * headDim + h * headDim;

        for (int d = 0; d < headDim; d++)
        {
            Atomic.Add(ref gradQ[qOffset + d], gs * k[kOffset + d]);
            Atomic.Add(ref gradK[kOffset + d], gs * q[qOffset + d]);
        }
    }
}

/// <summary>
/// Multi-Head Self-Attention без residual (residual в TransformerBlock).
/// Q, K, V проецируются через Wq, Wk, Wv, затем разбиваются на головы.
/// </summary>
public class MultiHeadAttentionLayer : ILayer
{
    private readonly Accelerator _accelerator;
    public readonly int EmbeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;

    internal LinearLayer _wq, _wk, _wv, _wo;

    // Буферы
    private MemoryBuffer1D<float, Stride1D.Dense> _inputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _qBuffer, _kBuffer, _vBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _scoresBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _attnOutputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradAttnWeightsBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradScoresBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradQBuffer, _gradKBuffer,
        _gradVBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradAttnOutBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradInputBuffer;

    // Cached views
    private ArrayView1D<float, Stride1D.Dense> _cachedInputView;
    private ArrayView1D<float, Stride1D.Dense> _cachedQView, _cachedKView, _cachedVView;
    private ArrayView1D<float, Stride1D.Dense> _cachedAttnWeightsView;
    private int _cachedSeqLen;

    // Kernel delegates
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>,
        SpecializedValue<float>> _scoresKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _softmaxKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _weightedSumKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _gradVKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>> _gradAttnWeightsKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>> _softmaxBackwardKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        int, SpecializedValue<int>, SpecializedValue<int>,
        SpecializedValue<float>> _gradQKKernel;

    private bool _disposed;

    public MultiHeadAttentionLayer(Accelerator accelerator,
        int embeddingDim, int numHeads = 4, int maxSeqLen = 80)
    {
        _accelerator = accelerator;
        EmbeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _maxSeqLen = maxSeqLen;

        if (embeddingDim % numHeads != 0)
            throw new ArgumentException(
                $"embeddingDim ({embeddingDim}) должен делиться на numHeads ({numHeads})");

        // Линейные проекции
        _wq = new LinearLayer(accelerator, embeddingDim, embeddingDim, false, maxSeqLen);
        _wk = new LinearLayer(accelerator, embeddingDim, embeddingDim, false, maxSeqLen);
        _wv = new LinearLayer(accelerator, embeddingDim, embeddingDim, false, maxSeqLen);
        _wo = new LinearLayer(accelerator, embeddingDim, embeddingDim, true, maxSeqLen);

        // Буферы
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
        _gradAttnOutBuffer = accelerator.Allocate1D<float>(maxInput);
        _gradInputBuffer = accelerator.Allocate1D<float>(maxInput);

        // Компиляция ядер
        _scoresKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>,
            SpecializedValue<float>>(MultiHeadAttentionKernels.MultiHeadScoresKernel);

        _softmaxKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadSoftmaxKernel);

        _weightedSumKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadWeightedSumKernel);

        _gradVKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadGradVKernel);

        _gradAttnWeightsKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadGradAttnWeightsKernel);

        _softmaxBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>>(
            MultiHeadAttentionKernels.MultiHeadSoftmaxBackwardKernel);

        _gradQKKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, SpecializedValue<int>, SpecializedValue<int>,
            SpecializedValue<float>>(MultiHeadAttentionKernels.MultiHeadGradQKKernel);
    }

    public string LayerType => "MultiHeadAttention";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        int inputSize = seqLen * EmbeddingDim;
        int scoresSize = _numHeads * seqLen * seqLen;
        _cachedSeqLen = seqLen;
        float scale = (float)Math.Sqrt(_headDim);

        // Кэшируем вход
        _inputBuffer.View.SubView(0, inputSize)
            .CopyFrom(input.SubView(0, inputSize));
        _cachedInputView = _inputBuffer.View.SubView(0, inputSize);

        // Проекции Q, K, V: [seqLen, embDim] → [seqLen, numHeads*headDim]
        var Q = _wq.Forward(_cachedInputView, seqLen);
        var K = _wk.Forward(_cachedInputView, seqLen);
        var V = _wv.Forward(_cachedInputView, seqLen);

        _qBuffer.View.SubView(0, inputSize).CopyFrom(Q);
        _kBuffer.View.SubView(0, inputSize).CopyFrom(K);
        _vBuffer.View.SubView(0, inputSize).CopyFrom(V);
        _cachedQView = _qBuffer.View.SubView(0, inputSize);
        _cachedKView = _kBuffer.View.SubView(0, inputSize);
        _cachedVView = _vBuffer.View.SubView(0, inputSize);

        // Attention scores: [numHeads, seqLen, seqLen]
        var scoresView = _scoresBuffer.View.SubView(0, scoresSize);
        _scoresKernel(scoresSize, Q, K, scoresView, seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim),
            SpecializedValue.New(scale));

        // Softmax по каждой строке каждой головы
        _softmaxKernel(_numHeads * seqLen, scoresView, seqLen,
            SpecializedValue.New(_numHeads));
        _cachedAttnWeightsView = scoresView;

        // Weighted sum: output[i,h,d] = Σ_j attn[h,i,j] * V[j,h,d]
        var attnOut = _attnOutputBuffer.View.SubView(0, inputSize);
        _weightedSumKernel(inputSize, scoresView, V, attnOut, seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim));

        // Output projection: Wo
        return _wo.Forward(attnOut, seqLen);
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr)
    {
        if (_cachedSeqLen == 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        int seqLen = _cachedSeqLen;
        int inputSize = seqLen * EmbeddingDim;
        int scoresSize = _numHeads * seqLen * seqLen;
        float scale = (float)Math.Sqrt(_headDim);

        // Backward через Wo
        var gradAttnOut = _wo.Backward(grads, lr);

        // gradV[j,h,d] = Σ_i attn[h,i,j] * gradAttnOut[i,h,d]
        var gradVView = _gradVBuffer.View.SubView(0, inputSize);
        _gradVKernel(inputSize, _cachedAttnWeightsView, gradAttnOut,
            gradVView, seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim));

        // gradAttnWeights[h,i,j] = Σ_d gradAttnOut[i,h,d] * V[j,h,d]
        var gradAttnWView = _gradAttnWeightsBuffer.View.SubView(0, scoresSize);
        _gradAttnWeightsKernel(scoresSize, gradAttnOut, _cachedVView,
            gradAttnWView, seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim));

        // Softmax backward
        var gradScoresView = _gradScoresBuffer.View.SubView(0, scoresSize);
        _softmaxBackwardKernel(_numHeads * seqLen,
            _cachedAttnWeightsView, gradAttnWView, gradScoresView,
            seqLen, SpecializedValue.New(_numHeads));

        // gradQ, gradK
        var gradQView = _gradQBuffer.View.SubView(0, inputSize);
        var gradKView = _gradKBuffer.View.SubView(0, inputSize);
        gradQView.MemSetToZero();
        gradKView.MemSetToZero();

        _gradQKKernel(scoresSize, gradScoresView,
            _cachedQView, _cachedKView, gradQView, gradKView,
            seqLen,
            SpecializedValue.New(_numHeads),
            SpecializedValue.New(_headDim),
            SpecializedValue.New(scale));

        // Backward через Wq, Wk, Wv
        var gradInputQ = _wq.Backward(gradQView, lr);
        var gradInputK = _wk.Backward(gradKView, lr);
        var gradInputV = _wv.Backward(gradVView, lr);

        // Суммируем градиенты от Q, K, V
        var gradInputView = _gradInputBuffer.View.SubView(0, inputSize);
        // Используем простой kernel для суммы трёх
        for (int idx = 0; idx < inputSize; idx++) { } // placeholder — используем GPU ядро:

        // Переиспользуем аккумуляцию
        gradInputView.CopyFrom(gradInputQ.SubView(0, inputSize));
        // gradInput += gradInputK
        var accumKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            (idx, dst, src) => { dst[idx] += src[idx]; });
        accumKernel(inputSize, gradInputView, gradInputK.SubView(0, inputSize));
        accumKernel(inputSize, gradInputView, gradInputV.SubView(0, inputSize));

        return gradInputView;
    }

    public int Parameters() =>
        _wq.Parameters() + _wk.Parameters() +
        _wv.Parameters() + _wo.Parameters();

    public void Dispose()
    {
        if (!_disposed)
        {
            _wq.Dispose(); _wk.Dispose();
            _wv.Dispose(); _wo.Dispose();
            _inputBuffer.Dispose();
            _qBuffer.Dispose(); _kBuffer.Dispose(); _vBuffer.Dispose();
            _scoresBuffer.Dispose(); _attnOutputBuffer.Dispose();
            _gradAttnWeightsBuffer.Dispose(); _gradScoresBuffer.Dispose();
            _gradQBuffer.Dispose(); _gradKBuffer.Dispose();
            _gradVBuffer.Dispose(); _gradAttnOutBuffer.Dispose();
            _gradInputBuffer.Dispose();
            _disposed = true;
        }
    }
}