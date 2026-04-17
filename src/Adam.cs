//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public class Adam : IDisposable
{
    private readonly float _beta1 = 0.9f;
    private readonly float _beta2 = 0.999f;
    private readonly float _epsilon = 1e-8f;
    private int _timestep;

    private readonly Accelerator _accelerator;
    private MemoryBuffer1D<float, Stride1D.Dense> _m;
    private MemoryBuffer1D<float, Stride1D.Dense> _v;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        float, float, float, float, float, float> _adamKernel;

    private bool _disposed;

    public Adam(Accelerator accelerator, int size)
    {
        _accelerator = accelerator;
        _timestep = 0;

        _m = accelerator.Allocate1D<float>(size);
        _v = accelerator.Allocate1D<float>(size);
        _m.MemSetToZero();
        _v.MemSetToZero();

        _adamKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
            float, float, float, float, float, float>(
            AdamKernels.AdamUpdateKernel);
    }

    public void Step(MemoryBuffer1D<float, Stride1D.Dense> paramsBuffer,
                     MemoryBuffer1D<float, Stride1D.Dense> gradsBuffer,
                     float lr)
    {
        _timestep++;
        float biasCorrection1 = 1.0f - MathF.Pow(_beta1, _timestep);
        float biasCorrection2 = 1.0f - MathF.Pow(_beta2, _timestep);

        _adamKernel(
            new Index1D((int)paramsBuffer.Length),
            paramsBuffer.View, gradsBuffer.View,
            _m.View, _v.View,
            lr, _beta1, _beta2, _epsilon,
            biasCorrection1, biasCorrection2);

        gradsBuffer.MemSetToZero();
    }

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
        float lr, float beta1, float beta2, float epsilon,
        float biasCorrection1, float biasCorrection2)
    {
        float g = gradients[index];
        float mOld = m[index];
        float vOld = v[index];

        float mNew = beta1 * mOld + (1.0f - beta1) * g;
        float vNew = beta2 * vOld + (1.0f - beta2) * g * g;

        float mHat = mNew / biasCorrection1;
        float vHat = vNew / biasCorrection2;

        parameters[index] -= lr * mHat / (XMath.Sqrt(vHat) + epsilon);
        m[index] = mNew;
        v[index] = vNew;
    }
}