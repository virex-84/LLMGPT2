//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class GELUKernels
{
    private const float Coeff = 0.044715f;
    private const float Sqrt2OverPi = 0.7978845608f;

    public static void GELUForwardKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        ArrayView1D<float, Stride1D.Dense> tanhCache)
    {
        float x = input[index];
        float inner = Sqrt2OverPi * (x + Coeff * x * x * x);
        float t = XMath.Tanh(inner);
        tanhCache[index] = t;
        output[index] = 0.5f * x * (1.0f + t);
    }

    public static void GELUBackwardKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> tanhCache,
        ArrayView1D<float, Stride1D.Dense> gradInput)
    {
        float x = input[index];
        float t = tanhCache[index];
        float tanhSq = t * t;
        float dGelu = 0.5f * (1.0f + t)
                     + 0.5f * x * (1.0f - tanhSq)
                     * Sqrt2OverPi * (1.0f + 3.0f * Coeff * x * x);
        gradInput[index] = gradOutput[index] * dGelu;
    }

    // ── НОВЫЙ: объединённый kernel W1→GELU за один проход ────────
    // Применяет GELU к уже вычисленному W1 output in-place.
    // Это позволяет избежать отдельного CopyFrom (GPU→GPU копии).
    // Используется когда W1.Forward возвращает view в _outputBuffer W1,
    // который мы можем использовать напрямую без копирования.
    public static void GELUInPlaceKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> data,     // W1 output (in-place)
        ArrayView1D<float, Stride1D.Dense> tanhCache,
        ArrayView1D<float, Stride1D.Dense> inputCache) // кэш pre-GELU для backward
    {
        float x = data[index];
        inputCache[index] = x;                        // сохраняем pre-GELU
        float inner = Sqrt2OverPi * (x + Coeff * x * x * x);
        float t = XMath.Tanh(inner);
        tanhCache[index] = t;
        data[index] = 0.5f * x * (1.0f + t);    // in-place GELU
    }
}

public class GELUFeedForwardLayer : ILayer
{
    private readonly Accelerator _accelerator;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;
    private readonly int _maxSeqLen;

    public LinearLayer W1 { get; }
    public LinearLayer W2 { get; }

    // ── Буферы ───────────────────────────────────────────────────
    // _hiddenPreGelu: кэш входа GELU (нужен для backward)
    // _tanhCache:     кэш tanh (нужен для backward без пересчёта)
    // _gradPreGelu:   буфер для градиента через GELU
    // _hiddenPostGelu УБРАН — теперь делаем GELU in-place в буфере W1
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _hiddenPreGelu;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tanhCache;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _gradPreGelu;

    private int _cachedSeqLen;
    // Кэшируем view результата W1 (используется в backward как gradHidden)
    private ArrayView1D<float, Stride1D.Dense> _cachedW1OutputView;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _geluForwardKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _geluBackwardKernel;

    // ── НОВЫЙ kernel: in-place GELU ───────────────────────────────
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> _geluInPlaceKernel;

    private bool _disposed;

    public GELUFeedForwardLayer(
        Accelerator accelerator,
        int embeddingDim,
        int hiddenDim,
        int maxSeqLen)
    {
        if (accelerator == null) throw new ArgumentNullException(nameof(accelerator));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _accelerator = accelerator;
        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _maxSeqLen = maxSeqLen;

        W1 = new LinearLayer(accelerator, embeddingDim, hiddenDim, true, maxSeqLen);
        W2 = new LinearLayer(accelerator, hiddenDim, embeddingDim, true, maxSeqLen);

        int maxHiddenSize = maxSeqLen * hiddenDim;

        // _hiddenPostGelu убран: GELU теперь in-place в буфере W1
        _hiddenPreGelu = accelerator.Allocate1D<float>(maxHiddenSize);
        _tanhCache = accelerator.Allocate1D<float>(maxHiddenSize);
        _gradPreGelu = accelerator.Allocate1D<float>(maxHiddenSize);

        _geluForwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(GELUKernels.GELUForwardKernel);

        _geluBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(GELUKernels.GELUBackwardKernel);

        _geluInPlaceKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(GELUKernels.GELUInPlaceKernel);
    }

    public string LayerType => "GELUFFN";

    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        if (seqLen <= 0 || seqLen > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(seqLen));

        _cachedSeqLen = seqLen;
        int hiddenSize = seqLen * _hiddenDim;

        // W1 forward: embeddingDim → hiddenDim
        // Результат живёт в _outputBuffer внутри W1
        var w1Out = W1.Forward(input, seqLen);

        // ── ОПТИМИЗАЦИЯ: GELU in-place, без CopyFrom ─────────────
        // Оригинал: preGeluView.CopyFrom(hiddenPre) → лишняя GPU→GPU копия
        // Новый код: GELUInPlaceKernel пишет в w1Out напрямую,
        // параллельно сохраняя pre-GELU в _hiddenPreGelu и tanh в _tanhCache
        var preGeluView = _hiddenPreGelu.View.SubView(0, hiddenSize);
        var tanhView = _tanhCache.View.SubView(0, hiddenSize);

        _geluInPlaceKernel(hiddenSize, w1Out, tanhView, preGeluView);

        // Сохраняем view для backward (w1Out теперь содержит post-GELU)
        _cachedW1OutputView = w1Out;

        // W2: hiddenDim → embeddingDim
        return W2.Forward(w1Out, seqLen);
    }

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

        // Gradient через GELU (кэшированный tanh — нет пересчёта)
        var preGeluView = _hiddenPreGelu.View.SubView(0, hiddenSize);
        var tanhView = _tanhCache.View.SubView(0, hiddenSize);
        var gradPreGelView = _gradPreGelu.View.SubView(0, hiddenSize);

        _geluBackwardKernel(
            hiddenSize,
            gradHidden.SubView(0, hiddenSize),
            preGeluView,
            tanhView,
            gradPreGelView);

        // Gradient через W1
        return W1.Backward(gradPreGelView, lr);
    }

    public int Parameters() => W1.Parameters() + W2.Parameters();

    public void Dispose()
    {
        if (!_disposed)
        {
            W1.Dispose();
            W2.Dispose();
            _hiddenPreGelu.Dispose();
            _tanhCache.Dispose();
            _gradPreGelu.Dispose();
            _disposed = true;
        }
    }
}