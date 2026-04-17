//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;

namespace LLM.ILGPU;

public class Context : IDisposable
{
    private readonly global::ILGPU.Context _context;
    private readonly Accelerator _accelerator;
    private bool _disposed;

    public Context()
    {
        _context = global::ILGPU.Context.Create(builder => builder.OpenCL());
        var devices = _context.GetCLDevices();
        if (devices.Count == 0)
            throw new InvalidOperationException(
                "OpenCL устройства не найдены.");

        _accelerator = devices[0].CreateAccelerator(_context);

        Console.WriteLine($"=== GPU INFO ===");
        Console.WriteLine($"Device: {_accelerator.Name}");
        Console.WriteLine($"Type: {_accelerator.AcceleratorType}");
        Console.WriteLine($"Memory: {_accelerator.MemorySize / (1024 * 1024)} MB");
        Console.WriteLine($"Max threads/group: {_accelerator.MaxNumThreadsPerGroup}");
        Console.WriteLine($"Warp size: {_accelerator.WarpSize}");
        Console.WriteLine("================\n");
    }

    public Accelerator Accelerator => _accelerator;

    public void Dispose()
    {
        if (!_disposed)
        {
            _accelerator.Dispose();
            _context.Dispose();
            _disposed = true;
        }
    }
}