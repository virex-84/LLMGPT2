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
        _context = global::ILGPU.Context.Create(builder =>
            builder
                .AllAccelerators()
                // ── Производительность ───────────────────────────
                // O2: дорогие трансформации — лучший код kernels
                .Optimize(OptimizationLevel.O2)

                // Aggressive: все функции инлайнятся
                // убирает overhead вызовов внутри kernel
                .Inlining(InliningMode.Aggressive)

                // Fast32BitOnly: все double → float автоматически
                // для нейросетей double не нужен, float быстрее
                // На AMD iGPU float в 2-4x быстрее double
                .Math(MathMode.Fast32BitOnly)

                // ── Параллельная генерация кода ──────────────────
                // Компилирует разные kernels параллельно на CPU
                // Ускоряет первый WarmUp
                .PageLocking(PageLockingMode.Auto)

                // ── Алгоритмы ────────────────────────────────────
                // Обязательно для XMath.Log/Exp/Sqrt
                .EnableAlgorithms()

                // ── Debug отключаем ──────────────────────────────
                // Без отладчика — никаких debug символов
                .DebugSymbols(DebugSymbolsMode.Disabled)
                );

        var devices = _context.Devices;

        if (devices.Length == 0)
            throw new InvalidOperationException("Устройства не найдены.");

        Console.WriteLine("Доступные устройства:");
        for (int i = 0; i < devices.Length; i++)
        {
            Console.WriteLine($"[{i}] {devices[i].AcceleratorType}: {devices[i].Name}");
        }

        int selectedIndex = -1;
        while (selectedIndex < 0)
        {
            Console.Write("\nВыберите номер устройства: ");
            string? input = Console.ReadLine();

            if (int.TryParse(input, out int idx)
                && idx >= 0
                && idx < devices.Length)
            {
                selectedIndex = idx;
            }
            else
            {
                Console.WriteLine(
                    $"  Некорректный выбор. Введите число от 0 до " +
                    $"{devices.Length - 1}.");
            }
        }

        _accelerator = devices[selectedIndex].CreateAccelerator(_context);

        devices[selectedIndex].PrintInformation(Console.Out);
        Console.WriteLine();
    }

    public Accelerator Accelerator => _accelerator;

    public void Dispose()
    {
        if (!_disposed)
        {
            _accelerator?.Dispose();
            _context?.Dispose();
            _disposed = true;
        }
    }
}