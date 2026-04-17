//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using System;

namespace LLM.ILGPU;

/// <summary>
/// Кернели для GELU активации.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// </summary>
public static class GELUKernels
{
    /// <summary>
    /// GELU forward pass.
    /// </summary>
    public static void GELUForwardKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output)
    {
        float x = input[index];
        // Approximation из оригинального GPT-2 / transformer libraries
        float coeff = 0.044715f;
        float sqrt2OverPi = 0.7978845608f; // sqrt(2/π)
        float inner = sqrt2OverPi * (x + coeff * x * x * x);
        float tanh = XMath.Tanh(inner);
        output[index] = 0.5f * x * (1.0f + tanh);
    }

    /// <summary>
    /// GELU backward pass.
    /// dy/dx = 0.5*(1+tanh) + 0.5*x*(1-tanh²)*coeff*(1+3*coeff*x²)
    /// </summary>
    public static void GELUBackwardKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> gradInput)
    {
        float x = input[index];
        float coeff = 0.044715f;
        float sqrt2OverPi = 0.7978845608f;
        float inner = sqrt2OverPi * (x + coeff * x * x * x);
        float tanh = XMath.Tanh(inner);
        float tanhSq = tanh * tanh;

        // Производная GELU
        float dGelu = 0.5f * (1.0f + tanh)
                    + 0.5f * x * (1.0f - tanhSq)
                    * sqrt2OverPi * (1.0f + 3.0f * coeff * x * x);

        gradInput[index] = gradOutput[index] * dGelu;
    }
}

/// <summary>
/// GPT-2 совместимый Feed-Forward слой с GELU активацией.
/// 
/// Формула: output = W2 · GELU(W1 · x + b1) + b2
/// 
/// Отличия от FeedForwardLayer (ReLU):
///   - GELU вместо ReLU (гладкая аппроксимация)
///   - Стандарт для GPT-2 / BERT / современных трансформеров
/// </summary>
public class GELUFeedForwardLayer : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _maxSeqLen;

    /// <summary>W1: embedding → hidden (с bias)</summary>
    public LinearLayer W1 { get; }

    /// <summary>W2: hidden → embedding (с bias)</summary>
    public LinearLayer W2 { get; }

    // Промежуточные буферы
    private MemoryBuffer1D<float, Stride1D.Dense> _hiddenPreGelu;
    private MemoryBuffer1D<float, Stride1D.Dense> _hiddenPostGelu;
    private MemoryBuffer1D<float, Stride1D.Dense> _cachedInput;
    private int _cachedSeqLen;

    // Кернелы
    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _geluForwardKernel;

    private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _geluBackwardKernel;

    private bool _disposed;

    public GELUFeedForwardLayer(Accelerator accelerator, int embeddingDim,
        int hiddenDim, int maxSeqLen)
    {
        _accelerator = accelerator;
        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _maxSeqLen = maxSeqLen;

        // GPT-2 FFN использует bias
        W1 = new LinearLayer(accelerator, embeddingDim, hiddenDim, true, maxSeqLen);
        W2 = new LinearLayer(accelerator, hiddenDim, embeddingDim, true, maxSeqLen);

        int hiddenSize = maxSeqLen * hiddenDim;
        _hiddenPreGelu = accelerator.Allocate1D<float>(hiddenSize);
        _hiddenPostGelu = accelerator.Allocate1D<float>(hiddenSize);
        _cachedInput = accelerator.Allocate1D<float>(maxSeqLen * embeddingDim);

        _geluForwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            GELUKernels.GELUForwardKernel);

        _geluBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            GELUKernels.GELUBackwardKernel);
    }

    public string LayerType => "GELUFFN";

    /// <summary>
    /// Прямой проход: x → W1 → GELU → W2 → output
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        _cachedSeqLen = seqLen;
        int inputSize = seqLen * _embeddingDim;

        // Кэшируем вход для backward
        _cachedInput.View.SubView(0, inputSize)
            .CopyFrom(input.SubView(0, inputSize));

        // W1: embedding → hidden
        var hiddenPre = W1.Forward(input, seqLen);
        int hiddenSize = seqLen * _hiddenDim;
        _hiddenPreGelu.View.SubView(0, hiddenSize)
            .CopyFrom(hiddenPre.SubView(0, hiddenSize));

        // GELU активация
        _geluForwardKernel(hiddenSize,
            _hiddenPreGelu.View.SubView(0, hiddenSize),
            _hiddenPostGelu.View.SubView(0, hiddenSize));

        // W2: hidden → embedding
        return W2.Forward(_hiddenPostGelu.View.SubView(0, hiddenSize), seqLen);
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
        int hiddenSize = seqLen * _hiddenDim;

        // Gradient через W2
        var gradHidden = W2.Backward(gradOutput, lr);

        // Gradient через GELU
        var gradPreGelu = _accelerator.Allocate1D<float>(hiddenSize);
        _geluBackwardKernel(hiddenSize,
            gradHidden,
            _hiddenPreGelu.View.SubView(0, hiddenSize),
            gradPreGelu.View);

        // Gradient через W1
        var gradInput = W1.Backward(gradPreGelu.View, lr);

        gradPreGelu.Dispose();
        return gradInput;
    }

    public int Parameters() => W1.Parameters() + W2.Parameters();

    public void Dispose()
    {
        if (!_disposed)
        {
            W1.Dispose(); W2.Dispose();
            _hiddenPreGelu.Dispose();
            _hiddenPostGelu.Dispose();
            _cachedInput.Dispose();
            _disposed = true;
        }
    }
}
