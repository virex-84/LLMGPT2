//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class ResidualKernels
{
    // ResidualAdd in-place: residual[i] += input[i], развёртка x4
    public static void ResidualAddKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> residual,
        ArrayView1D<float, Stride1D.Dense> input,
        SpecializedValue<int> totalSize)
    {
        int i = index * 4;
        int total = totalSize;

        if (i + 3 < total)
        {
            residual[i] += input[i];
            residual[i + 1] += input[i + 1];
            residual[i + 2] += input[i + 2];
            residual[i + 3] += input[i + 3];
        }
        else
        {
            for (int j = i; j < total; j++)
                residual[j] += input[j];
        }
    }
}

public class TransformerBlockPreNorm : ILayer, IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _maxSeqLen;

    internal readonly LayerNormLayer _ln1;
    internal readonly MultiHeadAttentionLayer _attention;
    internal readonly LayerNormLayer _ln2;
    internal readonly GELUFeedForwardLayer _ffn;

    // Минимальный набор буферов
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _residualBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _attnResidualBuffer;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _gradBuffer1;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _gradBuffer2;

    private int _cachedSeqLen;
    private bool _disposed;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _residualAddKernel;

    public int EmbeddingDim => _embeddingDim;
    public int HiddenDim => _hiddenDim;
    public int NumHeads => _numHeads;
    public string LayerType => "TransformerBlockPreNorm";

    public TransformerBlockPreNorm(
        Accelerator accelerator,
        int embeddingDim,
        int numHeads,
        int hiddenDim,
        int maxSeqLen)
    {
        _accelerator = accelerator;
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _hiddenDim = hiddenDim;
        _maxSeqLen = maxSeqLen;

        _ln1 = new LayerNormLayer(accelerator, embeddingDim, maxSeqLen);
        _attention = new MultiHeadAttentionLayer(
                         accelerator, embeddingDim, numHeads, maxSeqLen);
        _ln2 = new LayerNormLayer(accelerator, embeddingDim, maxSeqLen);
        _ffn = new GELUFeedForwardLayer(
                         accelerator, embeddingDim, hiddenDim, maxSeqLen);

        int bufferSize = maxSeqLen * embeddingDim;

        // Только необходимые буферы
        _residualBuffer = accelerator.Allocate1D<float>(bufferSize);
        _attnResidualBuffer = accelerator.Allocate1D<float>(bufferSize);
        _gradBuffer1 = accelerator.Allocate1D<float>(bufferSize);
        _gradBuffer2 = accelerator.Allocate1D<float>(bufferSize);

        _residualAddKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(
            ResidualKernels.ResidualAddKernel);
    }

    // ─────────────────────────────────────────────────────────────
    // Forward:
    //   residual1 = input                    (через .CopyFrom — DMA)
    //   x         = Attention(LN1(input))
    //   x         = residual1 + x            (ResidualAdd kernel)
    //   residual2 = x                        (через .CopyFrom — DMA)
    //   x         = FFN(LN2(x))
    //   output    = residual2 + x            (ResidualAdd kernel)
    // ─────────────────────────────────────────────────────────────
    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        _cachedSeqLen = seqLen;
        int totalDim = seqLen * _embeddingDim;
        int threads = (totalDim + 3) / 4;

        var residualSub = _residualBuffer.View.SubView(0, totalDim);
        var attnResSub = _attnResidualBuffer.View.SubView(0, totalDim);
        var inputSub = input.SubView(0, totalDim);

        // ── Attention Block ──────────────────────────────────────

        // residual1 = input (встроенный DMA-копировщик — быстрее kernel)
        residualSub.CopyFrom(inputSub);

        // LN1 → Attention
        var ln1Out = _ln1.Forward(inputSub, seqLen);
        var attnOut = _attention.Forward(ln1Out, seqLen);

        // residual1 + attnOut (in-place kernel)
        _residualAddKernel(threads, residualSub, attnOut,
            SpecializedValue.New(totalDim));

        // ── FFN Block ────────────────────────────────────────────

        // residual2 = residual1 + attnOut (DMA копия для Backward)
        attnResSub.CopyFrom(residualSub);

        // LN2 → FFN
        var ln2Out = _ln2.Forward(residualSub, seqLen);
        var ffnOut = _ffn.Forward(ln2Out, seqLen);

        // residual2 + ffnOut (in-place kernel)
        _residualAddKernel(threads, residualSub, ffnOut,
            SpecializedValue.New(totalDim));

        return residualSub;
    }

    // ─────────────────────────────────────────────────────────────
    // Backward:
    //   gradOutput → FFN.Backward → LN2.Backward
    //   + residual2 pass-through
    //   → Attention.Backward → LN1.Backward
    //   + residual1 pass-through
    // ─────────────────────────────────────────────────────────────
    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> gradOutput, float lr)
    {
        if (_cachedSeqLen <= 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        int seqLen = _cachedSeqLen;
        int totalDim = seqLen * _embeddingDim;
        int threads = (totalDim + 3) / 4;

        var grad1Sub = _gradBuffer1.View.SubView(0, totalDim);
        var grad2Sub = _gradBuffer2.View.SubView(0, totalDim);

        // ── FFN branch ──────────────────────────────────────────

        // Grad через FFN + LN2
        var gradAfterFfn = _ffn.Backward(gradOutput, lr);
        var gradAfterLn2 = _ln2.Backward(gradAfterFfn, lr);

        // Residual2: grad1 = gradOutput + gradAfterLn2
        grad1Sub.CopyFrom(gradOutput.SubView(0, totalDim));
        _residualAddKernel(threads, grad1Sub, gradAfterLn2,
            SpecializedValue.New(totalDim));

        // ── Attention branch ────────────────────────────────────

        // Grad через Attention + LN1
        var gradAfterAttn = _attention.Backward(grad1Sub, lr);
        var gradAfterLn1 = _ln1.Backward(gradAfterAttn, lr);

        // Residual1: grad2 = grad1 + gradAfterLn1
        grad2Sub.CopyFrom(grad1Sub);
        _residualAddKernel(threads, grad2Sub, gradAfterLn1,
            SpecializedValue.New(totalDim));

        return grad2Sub;
    }

    public int Parameters() =>
        _ln1.Parameters() +
        _attention.Parameters() +
        _ln2.Parameters() +
        _ffn.Parameters();

    public void Dispose()
    {
        if (!_disposed)
        {
            _ln1.Dispose();
            _attention.Dispose();
            _ln2.Dispose();
            _ffn.Dispose();
            _residualBuffer.Dispose();
            _attnResidualBuffer.Dispose();
            _gradBuffer1.Dispose();
            _gradBuffer2.Dispose();
            _disposed = true;
        }
    }
}