//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class MatrixOps
{
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<float>,
        ArrayView1D<uint, Stride1D.Dense>>? _cachedRandomNormalInitKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _cachedReluKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _cachedReluGradKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int>? _cachedSoftmaxKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<int, Stride1D.Dense>, int>? _cachedGreedyDecodeKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _cachedResidualAddKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>? _cachedNormKernel;
    private static Action<Index1D, ArrayView1D<float, Stride1D.Dense>,
        float>? _cachedScaleGradsKernel;

    public static void ZeroInit(Accelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> buffer)
    {
        buffer.MemSetToZero();
    }

    public static void RandomNormalInit(Accelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> buffer, float std)
    {
        _cachedRandomNormalInitKernel ??=
            accelerator.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<float, Stride1D.Dense>, SpecializedValue<float>,
                ArrayView1D<uint, Stride1D.Dense>>(RandomNormalInitKernel);

        var seedBuffer = accelerator.Allocate1D(
            new uint[] { (uint)DateTime.Now.Millisecond });
        _cachedRandomNormalInitKernel(new Index1D((int)buffer.Length),
            buffer.View, SpecializedValue.New(std), seedBuffer.View);
        seedBuffer.Dispose();
    }

    public static void Softmax(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> input,
        MemoryBuffer1D<float, Stride1D.Dense> output, int rows, int cols)
    {
        _cachedSoftmaxKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, int>(
            (rowIdx, inView, outView, c) =>
            {
                int offset = rowIdx * c;
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < c; j++)
                    maxVal = XMath.Max(maxVal, inView[offset + j]);

                float sum = 0;
                for (int j = 0; j < c; j++)
                {
                    float e = XMath.Exp(inView[offset + j] - maxVal);
                    outView[offset + j] = e;
                    sum += e;
                }
                for (int j = 0; j < c; j++)
                    outView[offset + j] /= sum;
            });
        _cachedSoftmaxKernel(rows, input, output.View, cols);
    }

    public static void ClipGradients(Accelerator accelerator,
        ArrayView1D<float, Stride1D.Dense> grads, float maxNorm,
        MemoryBuffer1D<float, Stride1D.Dense> normBuffer)
    {
        normBuffer.MemSetToZero();

        _cachedNormKernel ??= accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(
            (index, g, normBuf) =>
            {
                float val = g[index];
                Atomic.Add(ref normBuf[0], val * val);
            });

        _cachedNormKernel((Index1D)grads.Length, grads, normBuffer.View);

        float[] normHost = new float[1];
        normBuffer.CopyToCPU(normHost);
        float totalNorm = (float)Math.Sqrt(normHost[0]);

        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / (totalNorm + 1e-6f);
            _cachedScaleGradsKernel ??=
                accelerator.LoadAutoGroupedStreamKernel<Index1D,
                    ArrayView1D<float, Stride1D.Dense>, float>(
                    (idx, g, s) => { g[idx] *= s; });
            _cachedScaleGradsKernel((int)grads.Length, grads, scale);
        }
    }

    /// <summary>
    /// ✅ Детерминированный хэш от индекса — нет race condition на seed[0].
    /// </summary>
    public static void RandomNormalInitKernel(Index1D index,
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
        float u1 = (s1 % 10000 + 1) / 10000.0f;

        uint s2 = baseSeed + (uint)index * 67890u + 2u;
        s2 = (s2 ^ 61u) ^ (s2 >> 16);
        s2 += s2 << 3;
        s2 ^= s2 >> 4;
        s2 *= 0x27d4eb2du;
        s2 ^= s2 >> 15;
        float u2 = (s2 % 10000 + 1) / 10000.0f;

        float z0 = (float)Math.Sqrt(-2.0 * Math.Log(u1)) *
                    (float)Math.Cos(2.0 * 3.14159265359 * u2);
        output[index] = stdDev * z0;
    }
}