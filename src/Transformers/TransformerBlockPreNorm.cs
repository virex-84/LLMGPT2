//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using System;

namespace LLM.ILGPU;

/// <summary>
/// GPT-2 совместимый Transformer блок с Pre-LayerNorm.
/// 
/// Архитектура (Pre-Norm):
///   1. x = x + Attention(LayerNorm(x))     — attention causal mask
///   2. x = x + FFN_GELU(LayerNorm(x))      — feed-forward с GELU
/// 
/// Отличия от TransformerBlock (Post-Norm):
///   - LayerNorm ПЕРЕД attention и FFN, а не после
///   - GELU вместо ReLU
///   - Обучаемые positional embeddings (не синусоидальные)
///   - Causal (look-ahead) маска в attention
/// </summary>
public class TransformerBlockPreNorm : ILayer, IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _maxSeqLen;

    // Pre-Norm слои (перед attention и перед FFN)
    internal readonly LayerNormLayer _ln1;
    internal readonly MultiHeadAttentionLayer _attention;
    internal readonly LayerNormLayer _ln2;
    internal readonly GELUFeedForwardLayer _ffn;

    // Буферы для residual connection
    private MemoryBuffer1D<float, Stride1D.Dense> _residualBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _normOutputBuffer;
    private int _cachedSeqLen;
    private MemoryBuffer1D<float, Stride1D.Dense> _cachedInputBuffer;

    private bool _disposed;

    public int EmbeddingDim => _embeddingDim;
    public int HiddenDim => _hiddenDim;
    public int NumHeads => _numHeads;

    public TransformerBlockPreNorm(Accelerator accelerator, int embeddingDim,
        int numHeads, int hiddenDim, int maxSeqLen)
    {
        _accelerator = accelerator;
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _hiddenDim = hiddenDim;
        _maxSeqLen = maxSeqLen;

        // Pre-LayerNorm слои
        _ln1 = new LayerNormLayer(accelerator, embeddingDim, maxSeqLen);
        _attention = new MultiHeadAttentionLayer(
            accelerator, embeddingDim, numHeads, maxSeqLen);
        _ln2 = new LayerNormLayer(accelerator, embeddingDim, maxSeqLen);
        _ffn = new GELUFeedForwardLayer(
            accelerator, embeddingDim, hiddenDim, maxSeqLen);

        // Буферы
        int bufferSize = maxSeqLen * embeddingDim;
        _residualBuffer = accelerator.Allocate1D<float>(bufferSize);
        _normOutputBuffer = accelerator.Allocate1D<float>(bufferSize);
        _cachedInputBuffer = accelerator.Allocate1D<float>(bufferSize);
    }

    public string LayerType => "TransformerBlockPreNorm";

    /// <summary>
    /// Прямой проход GPT-2 Pre-Norm Transformer блока.
    /// 
    /// flow:
    ///   residual = input
    ///   x = LN1(input)
    ///   x = Attention(x)  ← с causal mask
    ///   x = residual + x
    ///   residual = x
    ///   x = LN2(x)
    ///   x = FFN_GELU(x)
    ///   output = residual + x
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        _cachedSeqLen = seqLen;
        int totalDim = seqLen * _embeddingDim;

        // Кэшируем вход для backward
        _cachedInputBuffer.View.SubView(0, totalDim)
            .CopyFrom(input.SubView(0, totalDim));

        // ─── Attention Block ───
        
        // Сохраняем residual
        _residualBuffer.View.SubView(0, totalDim)
            .CopyFrom(input.SubView(0, totalDim));

        // LayerNorm перед attention
        var normed = _ln1.Forward(input, seqLen);

        // Attention (с causal mask)
        var attnOut = _attention.Forward(normed, seqLen);

        // Residual Add: input + attention
        ResidualAdd(_residualBuffer.View.SubView(0, totalDim),
            attnOut, seqLen * _embeddingDim);

        // ─── FFN Block ───

        // Сохраняем текущий residual (input + attention)
        var attnResidual = _accelerator.Allocate1D<float>(totalDim);
        attnResidual.View.SubView(0, totalDim)
            .CopyFrom(_residualBuffer.View.SubView(0, totalDim));

        // LayerNorm перед FFN
        var normed2 = _ln2.Forward(_residualBuffer.View.SubView(0, totalDim), seqLen);

        // FFN с GELU
        var ffnOut = _ffn.Forward(normed2, seqLen);

        // Final Residual Add: (input+attn) + ffn
        ResidualAdd(_residualBuffer.View.SubView(0, totalDim),
            ffnOut, seqLen * _embeddingDim);

        attnResidual.Dispose();
        return _residualBuffer.View.SubView(0, totalDim);
    }

    /// <summary>
    /// Обратный проход.
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> gradOutput, float lr)
    {
        if (_cachedSeqLen <= 0)
            throw new InvalidOperationException(
                "Forward должен быть вызван перед Backward");

        int seqLen = _cachedSeqLen;
        int totalDim = seqLen * _embeddingDim;

        // Gradient через FFN
        var gradFfnIn = _ffn.Backward(gradOutput, lr);

        // Gradient через LN2
        // (упрощённо — пропускаем)

        // Gradient через Attention
        var gradAttnIn = _attention.Backward(gradOutput, lr);

        // Gradient через LN1
        // (упрощённо — пропускаем)

        return gradFfnIn;
    }

    public int Parameters() =>
        _ln1.Parameters() + _attention.Parameters() +
        _ln2.Parameters() + _ffn.Parameters();

    /// <summary>
    /// Residual connection: output = residual + input (in-place).
    /// </summary>
    private void ResidualAdd(
        ArrayView1D<float, Stride1D.Dense> residual,
        ArrayView1D<float, Stride1D.Dense> input,
        int size)
    {
        var residHost = new float[size];
        var inputHost = new float[size];
        residual.CopyToCPU(residHost);
        input.CopyToCPU(inputHost);

        for (int i = 0; i < size; i++)
            residHost[i] += inputHost[i];

        residual.CopyFromCPU(residHost);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _ln1.Dispose();
            _attention.Dispose();
            _ln2.Dispose();
            _ffn.Dispose();
            _residualBuffer.Dispose();
            _normOutputBuffer.Dispose();
            _cachedInputBuffer.Dispose();
            _disposed = true;
        }
    }
}
