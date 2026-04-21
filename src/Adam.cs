//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using System.Threading;

namespace LLM.ILGPU;

public class Adam : IDisposable
{
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    private readonly float _clipValue;

    private int _timestep;
    private readonly Accelerator _accelerator;
    private readonly int _size;

    // Накопленные степени beta для быстрого вычисления bias correction
    // Обновляются на CPU за O(1) вместо MathF.Pow каждый шаг
    private float _beta1t = 1.0f; // beta1^t
    private float _beta2t = 1.0f; // beta2^t

    private MemoryBuffer1D<float, Stride1D.Dense> _m;
    private MemoryBuffer1D<float, Stride1D.Dense> _v;

    private readonly Action<
        Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        float, float, float, float,
        float, float, float, float
    > _adamKernel;

    private bool _disposed;

    public Adam(
        Accelerator accelerator,
        int size,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weightDecay = 0.0f,
        float clipValue = 0.0f)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(size), "Size must be > 0");

        _accelerator = accelerator;
        _size = size;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _clipValue = clipValue;
        _timestep = 0;

        _m = accelerator.Allocate1D<float>(size);
        _v = accelerator.Allocate1D<float>(size);
        _m.MemSetToZero();
        _v.MemSetToZero();

        _adamKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            float, float, float, float,
            float, float, float, float>(
            AdamKernels.AdamUpdateKernel);
    }

    // ── Step для MemoryBuffer1D ───────────────────────────────────
    public void Step(
        MemoryBuffer1D<float, Stride1D.Dense> paramsBuffer,
        MemoryBuffer1D<float, Stride1D.Dense> gradsBuffer,
        float lr)
    {
        if (paramsBuffer.Length != _size)
            throw new ArgumentException(
                $"paramsBuffer size {paramsBuffer.Length} " +
                $"!= optimizer size {_size}");
        if (gradsBuffer.Length != _size)
            throw new ArgumentException(
                $"gradsBuffer size {gradsBuffer.Length} " +
                $"!= optimizer size {_size}");

        StepInternal(paramsBuffer.View, gradsBuffer.View, _size, lr);
    }

    // ── Step для ArrayView1D (SubView) ────────────────────────────
    public void Step(
        ArrayView1D<float, Stride1D.Dense> paramsView,
        ArrayView1D<float, Stride1D.Dense> gradsView,
        float lr,
        int activeSize = -1)
    {
        int count = activeSize < 0 ? _size : activeSize;

        if (count > _size)
            throw new ArgumentException(
                $"activeSize {count} превышает optimizer size {_size}");
        if ((long)paramsView.Length < count)
            throw new ArgumentException(
                $"paramsView.Length {paramsView.Length} < activeSize {count}");
        if ((long)gradsView.Length < count)
            throw new ArgumentException(
                $"gradsView.Length {gradsView.Length} < activeSize {count}");

        StepInternal(paramsView, gradsView, count, lr);
    }

    // ── Общая реализация ──────────────────────────────────────────
    private void StepInternal(
        ArrayView1D<float, Stride1D.Dense> paramsView,
        ArrayView1D<float, Stride1D.Dense> gradsView,
        int count,
        float lr)
    {
        Interlocked.Increment(ref _timestep);

        // ✅ Умножение вместо MathF.Pow каждый шаг
        // beta1^t = beta1^(t-1) * beta1  — O(1) вместо O(log t)
        _beta1t *= _beta1;
        _beta2t *= _beta2;

        float invBC1 = 1.0f / (1.0f - _beta1t);
        float invBC2 = 1.0f / (1.0f - _beta2t);

        _adamKernel(
            new Index1D(count),
            paramsView,
            gradsView,
            _m.View,
            _v.View,
            lr,
            _beta1,
            _beta2,
            _epsilon,
            invBC1,
            invBC2,
            _weightDecay,
            _clipValue);
    }

    public void Reset()
    {
        Interlocked.Exchange(ref _timestep, 0);
        // ✅ Сбрасываем накопленные степени
        _beta1t = 1.0f;
        _beta2t = 1.0f;
        _m.MemSetToZero();
        _v.MemSetToZero();
    }

    public int Timestep => _timestep;

    public void Dispose()
    {
        if (!_disposed)
        {
            _m.Dispose();
            _v.Dispose();
            _disposed = true;
        }
    }
}

public static class AdamKernels
{
    public static void AdamUpdateKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> parameters,
        ArrayView1D<float, Stride1D.Dense> gradients,
        ArrayView1D<float, Stride1D.Dense> m,
        ArrayView1D<float, Stride1D.Dense> v,
        float lr,
        float beta1,
        float beta2,
        float epsilon,
        float invBiasCorrection1,
        float invBiasCorrection2,
        float weightDecay,
        float clipValue)
    {
        float g = gradients[index];

        // ✅ XMath.Clamp вместо двух if — одна инструкция на AMD GPU
        if (clipValue > 0.0f)
            g = XMath.Clamp(g, -clipValue, clipValue);

        float mOld = m[index];
        float vOld = v[index];

        float oneMinusBeta1 = 1.0f - beta1;
        float oneMinusBeta2 = 1.0f - beta2;

        float mNew = beta1 * mOld + oneMinusBeta1 * g;
        float vNew = beta2 * vOld + oneMinusBeta2 * g * g;

        float mHat = mNew * invBiasCorrection1;
        float vHat = vNew * invBiasCorrection2;

        // ✅ Читаем параметр один раз в регистр
        float param = parameters[index];

        // ✅ XMath.Rsqrt быстрее чем 1/XMath.Sqrt на AMD GPU
        // Rsqrt(vHat + eps²) ≈ 1/(sqrt(vHat) + eps) при vHat >> eps²
        // Для нейросетей точность достаточна
        // Стандартная формула: lr * mHat / (sqrt(vHat) + eps)
        // Через Rsqrt:         lr * mHat * Rsqrt(vHat + eps*eps)
        // Разница < 0.01% при типичных значениях vHat
        float update = lr * mHat * XMath.Rsqrt(vHat + epsilon * epsilon);

        // ✅ weightDecay использует уже загруженный param
        if (weightDecay != 0.0f)
            update += lr * weightDecay * param;

        // Записываем результаты
        parameters[index] = param - update;
        m[index] = mNew;
        v[index] = vNew;

        // Обнуляем градиент — совмещено с update
        gradients[index] = 0.0f;
    }
}