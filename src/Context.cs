//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace LLM.ILGPU;

public class Context : IDisposable
{
    private readonly global::ILGPU.Context _context;
    private readonly Accelerator _accelerator;
    private bool _disposed;

    public Context()
    {
        var _context = global::ILGPU.Context.Create(builder => builder.AllAccelerators());

        Console.WriteLine("Доступные устройства:");
        var devices = _context.Devices;

        if (devices.Length == 0)
            throw new InvalidOperationException(
                "OpenCL устройства не найдены.");

        for (int i = 0; i < devices.Length; i++)
        {
            Console.WriteLine($"[{i}] {devices[i].AcceleratorType}: {devices[i].Name}");
        }

    choice:
        Console.Write("\nВыберите номер устройства: ");
        if (int.TryParse(Console.ReadLine(), out int choice) && choice >= 0 && choice < devices.Length)
        {
            var selectedDevice = devices[choice];

            // 5. Создаем акселератор на основе выбранного устройства
            _accelerator = selectedDevice.CreateAccelerator(_context);
        }
        else
        {
            Console.WriteLine("Некорректный выбор.");
            goto choice;
        }

        devices[choice].PrintInformation(Console.Out);
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