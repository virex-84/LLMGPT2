//https://github.com/virex-84

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace LLM.ILGPU;

public static class MatrixOps
{
    // ── Кэш per-accelerator через слабые ссылки ───────────────────
    private static readonly
        ConditionalWeakTable<Accelerator, MatrixOpsKernelCache> _kernelCaches = new();

    private static MatrixOpsKernelCache GetCache(Accelerator accelerator)
        => _kernelCaches.GetOrCreateValue(accelerator);

    // ── Переиспользуемый CPU-буфер для normHost (избегаем new float[1] каждый вызов) ──
    // НЕ thread-safe если несколько потоков на одном accelerator,
    // но для single-threaded training это ок.
    [ThreadStatic]
    private static float[]? _normHostBuffer;

    private static float[] NormHostBuffer
        => _normHostBuffer ??= new float[1];

    // ─────────────────────────────────────────────────────────────
    public static void ZeroInit(Accelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        buffer.MemSetToZero();
    }

    // ─────────────────────────────────────────────────────────────
    public static void RandomNormalInit(Accelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> buffer, float std)
    {
        var cache = GetCache(accelerator);
        cache.EnsureRandomNormalInit(accelerator);

        uint seed = (uint)(DateTime.Now.Ticks & 0xFFFFFFFF);
        using var seedBuffer = accelerator.Allocate1D(new uint[] { seed });

        cache.RandomNormalInitKernel!(
            new Index1D((int)buffer.Length),
            buffer.View,
            SpecializedValue.New(std),
            seedBuffer.View);

        accelerator.Synchronize();
    }

    // ─────────────────────────────────────────────────────────────
    public static void Softmax(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> input,
        MemoryBuffer1D<float, Stride1D.Dense> output,
        int rows, int cols)
    {
        var cache = GetCache(accelerator);
        cache.EnsureSoftmax(accelerator);
        cache.SoftmaxKernel!(rows, input, output.View, cols);
    }

    // ─────────────────────────────────────────────────────────────
    // ClipGradients — оптимизирован для gfx1150:
    // Используем двухпроходный reduce вместо Atomic.Add на каждый элемент.
    // Это критично для OpenCL — атомики медленнее чем на CUDA.
    // ─────────────────────────────────────────────────────────────
    public static void ClipGradients(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> grads,
        float maxNorm,
        MemoryBuffer1D<float, Stride1D.Dense> normBuffer)
    {
        var cache = GetCache(accelerator);
        cache.EnsureClipGradients(accelerator);

        int len = (int)grads.Length;

        // ── Двухпроходный reduce для нормы ──────────────────────
        // Шаг 1: каждый поток считает частичную сумму квадратов
        // в блоке размером groupSize=256 (= Max threads/group для gfx1150)
        const int groupSize = 256;
        int numGroups = (len + groupSize - 1) / groupSize;

        // Буфер под частичные суммы (переиспользуем normBuffer только если
        // numGroups==1, иначе нужен отдельный буфер)
        using var partialSums = accelerator.Allocate1D<float>(numGroups);
        partialSums.MemSetToZero();

        // Шаг 1: reduce по блокам — один Atomic.Add на блок, не на элемент
        cache.PartialNormKernel!(
            new Index1D(len),
            grads,
            partialSums.View,
            len,
            groupSize);

        // Шаг 2: суммируем частичные суммы (маленький массив → быстро)
        normBuffer.MemSetToZero();
        cache.FinalReduceKernel!(
            new Index1D(numGroups),
            partialSums.View,
            normBuffer.View);

        // CopyToCPU — неизбежно (нужно знать норму для scale),
        // но делаем это ПОСЛЕ всех GPU операций
        normBuffer.CopyToCPU(NormHostBuffer);
        float totalNorm = XMath.Sqrt(NormHostBuffer[0]);

        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / (totalNorm + 1e-6f);
            // Исправлен тип: Index1D вместо (int) — баг в оригинале
            cache.ScaleGradsKernel!(new Index1D(len), grads, scale);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // ScaleGradients — для gradient accumulation в батч-обучении
    // ─────────────────────────────────────────────────────────────
    public static void ScaleGradients(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> grads,
        float scale)
    {
        var cache = GetCache(accelerator);
        cache.EnsureClipGradients(accelerator);
        cache.ScaleGradsKernel!(new Index1D((int)grads.Length), grads, scale);
    }

    // ═══════════════════════════════════════════════════════════════════
    // GPU KERNELS
    // ═══════════════════════════════════════════════════════════════════

    public static void RandomNormalInitKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> output,
        SpecializedValue<float> stdDev,
        ArrayView1D<uint, Stride1D.Dense> seed)
    {
        uint baseSeed = seed[0];

        uint s1 = baseSeed + (uint)index * 12345u + 1u;
        s1 = (s1 ^ 61u) ^ (s1 >> 16);
        s1 += s1 << 3;
        s1 ^= s1 >> 4;
        s1 *= 0x27d4eb2du;
        s1 ^= s1 >> 15;
        float u1 = (float)(s1 % 999983u + 1u) / 999984.0f;

        uint s2 = baseSeed + (uint)index * 67890u + 2u;
        s2 = (s2 ^ 61u) ^ (s2 >> 16);
        s2 += s2 << 3;
        s2 ^= s2 >> 4;
        s2 *= 0x27d4eb2du;
        s2 ^= s2 >> 15;
        float u2 = (float)(s2 % 999983u + 1u) / 999984.0f;

        float z0 = XMath.Sqrt(-2.0f * XMath.Log(u1))
                 * XMath.Cos(2.0f * 3.14159265359f * u2);
        output[index] = stdDev * z0;
    }

    // ─────────────────────────────────────────────────────────────
    // Softmax — оригинальный, уже хорошо оптимизирован (x4 unroll)
    // ─────────────────────────────────────────────────────────────
    public static void SoftmaxKernelImpl(
        Index1D rowIdx,
        ArrayView1D<float, Stride1D.Dense> inView,
        ArrayView1D<float, Stride1D.Dense> outView,
        int c)
    {
        int offset = rowIdx * c;

        // Проход 1: max
        float maxVal = float.NegativeInfinity;
        for (int j = 0; j < c; j++)
            maxVal = XMath.Max(maxVal, inView[offset + j]);

        // Проход 2: exp + sum, развёртка x4
        float sum0 = 0f, sum1 = 0f, sum2 = 0f, sum3 = 0f;
        int k = 0;
        int limit4 = c - (c % 4);

        for (; k < limit4; k += 4)
        {
            float e0 = XMath.Exp(inView[offset + k] - maxVal);
            float e1 = XMath.Exp(inView[offset + k + 1] - maxVal);
            float e2 = XMath.Exp(inView[offset + k + 2] - maxVal);
            float e3 = XMath.Exp(inView[offset + k + 3] - maxVal);
            outView[offset + k] = e0;
            outView[offset + k + 1] = e1;
            outView[offset + k + 2] = e2;
            outView[offset + k + 3] = e3;
            sum0 += e0; sum1 += e1; sum2 += e2; sum3 += e3;
        }

        float sum = sum0 + sum1 + sum2 + sum3;
        for (; k < c; k++)
        {
            float e = XMath.Exp(inView[offset + k] - maxVal);
            outView[offset + k] = e;
            sum += e;
        }

        // Проход 3: нормировка, развёртка x4
        float invSum = 1.0f / sum;
        k = 0;
        for (; k < limit4; k += 4)
        {
            outView[offset + k] *= invSum;
            outView[offset + k + 1] *= invSum;
            outView[offset + k + 2] *= invSum;
            outView[offset + k + 3] *= invSum;
        }
        for (; k < c; k++)
            outView[offset + k] *= invSum;
    }

    // ─────────────────────────────────────────────────────────────
    // PartialNormKernel — блочный reduce для нормы градиентов.
    // Каждый поток обрабатывает свой элемент, один Atomic.Add
    // на блок (groupSize элементов) вместо одного на элемент.
    // Для gfx1150 с 256 потоками/группой это даёт ~256x меньше атомиков.
    // ─────────────────────────────────────────────────────────────
    public static void PartialNormKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> g,
        ArrayView1D<float, Stride1D.Dense> partialSums,
        int totalLen,
        int groupSize)
    {
        if (index >= totalLen) return;

        float val = g[index];
        int groupIdx = index / groupSize;

        // Atomic.Add на группу, а не на глобальный буфер —
        // в groupSize раз меньше конкуренции за атомик
        Atomic.Add(ref partialSums[groupIdx], val * val);
    }

    // ─────────────────────────────────────────────────────────────
    // FinalReduceKernel — суммирует частичные суммы блоков.
    // Запускается с numGroups потоков (обычно < 1000).
    // ─────────────────────────────────────────────────────────────
    public static void FinalReduceKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> partialSums,
        ArrayView1D<float, Stride1D.Dense> result)
    {
        Atomic.Add(ref result[0], partialSums[index]);
    }

    // ─────────────────────────────────────────────────────────────
    // NormKernelImpl — оставлен для совместимости (не используется
    // в новом ClipGradients, но может понадобиться снаружи)
    // ─────────────────────────────────────────────────────────────
    public static void NormKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> g,
        ArrayView1D<float, Stride1D.Dense> normBuf)
    {
        float val = g[index];
        Atomic.Add(ref normBuf[0], val * val);
    }

    public static void ScaleGradsKernelImpl(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> g,
        float scale)
    {
        g[idx] *= scale;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Кэш kernel-ов per-accelerator
// ═══════════════════════════════════════════════════════════════════
internal sealed class MatrixOpsKernelCache
{
    public Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<float>,
        ArrayView1D<uint, Stride1D.Dense>>? RandomNormalInitKernel;

    public Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int>? SoftmaxKernel;

    // ── Старый NormKernel (совместимость) ──────────────────────
    public Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? NormKernel;

    // ── Новые kernels для двухпроходного reduce ─────────────────
    public Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int,
        int>? PartialNormKernel;

    public Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? FinalReduceKernel;

    public Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        float>? ScaleGradsKernel;

    // ── Ленивая инициализация ───────────────────────────────────
    public void EnsureRandomNormalInit(Accelerator acc)
    {
        RandomNormalInitKernel ??= acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<float>,
            ArrayView1D<uint, Stride1D.Dense>>(
            MatrixOps.RandomNormalInitKernel);
    }

    public void EnsureSoftmax(Accelerator acc)
    {
        SoftmaxKernel ??= acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(MatrixOps.SoftmaxKernelImpl);
    }

    public void EnsureClipGradients(Accelerator acc)
    {
        // Старый kernel — для совместимости
        NormKernel ??= acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            MatrixOps.NormKernelImpl);

        // Новые — двухпроходный reduce
        PartialNormKernel ??= acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int,
            int>(MatrixOps.PartialNormKernelImpl);

        FinalReduceKernel ??= acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            MatrixOps.FinalReduceKernelImpl);

        ScaleGradsKernel ??= acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float>(MatrixOps.ScaleGradsKernelImpl);
    }
}