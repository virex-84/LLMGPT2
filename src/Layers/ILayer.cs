//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;

namespace LLM.ILGPU;

public interface ILayer : IDisposable
{
    string LayerType { get; }
    ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen);
    ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> grads, float lr);
    int Parameters();
}