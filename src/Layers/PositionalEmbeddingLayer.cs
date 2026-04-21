//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace LLM.ILGPU;

public static class PositionalEmbeddingKernels
{
    // ─────────────────────────────────────────────────────────────
    // Forward: output[i] = input[i] + posWeights[i]
    // Один поток на элемент, развёртка x4
    // totalSize = seqLen * embeddingDim
    // ─────────────────────────────────────────────────────────────
    public static void AddPositionalKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> posWeights,
        ArrayView1D<float, Stride1D.Dense> output,
        SpecializedValue<int> totalSize)
    {
        int i = index * 4;
        int total = totalSize;

        if (i + 3 < total)
        {
            output[i] = input[i] + posWeights[i];
            output[i + 1] = input[i + 1] + posWeights[i + 1];
            output[i + 2] = input[i + 2] + posWeights[i + 2];
            output[i + 3] = input[i + 3] + posWeights[i + 3];
        }
        else
        {
            // хвост
            for (int j = i; j < total; j++)
                output[j] = input[j] + posWeights[j];
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Backward: gradInput[i] = gradOutput[i]  (identity)
    //           gradPos[i]  += gradOutput[i]
    // Совмещаем оба в одном ядре, развёртка x4
    // ─────────────────────────────────────────────────────────────
    public static void PosEmbBackwardKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> gradOutput,
        ArrayView1D<float, Stride1D.Dense> gradInput,
        ArrayView1D<float, Stride1D.Dense> gradPosWeights,
        SpecializedValue<int> totalSize)
    {
        int i = index * 4;
        int total = totalSize;

        if (i + 3 < total)
        {
            float g0 = gradOutput[i];
            float g1 = gradOutput[i + 1];
            float g2 = gradOutput[i + 2];
            float g3 = gradOutput[i + 3];

            gradInput[i] = g0;
            gradInput[i + 1] = g1;
            gradInput[i + 2] = g2;
            gradInput[i + 3] = g3;

            // gradPos аккумулируется через Atomic (несколько батчей могут
            // писать одновременно; при одном батче — просто запись, Atomic безопасен)
            Atomic.Add(ref gradPosWeights[i], g0);
            Atomic.Add(ref gradPosWeights[i + 1], g1);
            Atomic.Add(ref gradPosWeights[i + 2], g2);
            Atomic.Add(ref gradPosWeights[i + 3], g3);
        }
        else
        {
            for (int j = i; j < total; j++)
            {
                float g = gradOutput[j];
                gradInput[j] = g;
                Atomic.Add(ref gradPosWeights[j], g);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
public class PositionalEmbeddingLayer : ILayer, IDisposable
{
    private readonly Accelerator _accelerator;
    public int MaxSeqLen { get; }
    public int EmbeddingDim { get; }

    public MemoryBuffer1D<float, Stride1D.Dense> PositionWeights;
    public MemoryBuffer1D<float, Stride1D.Dense> GradPositionWeights;

    // ── Pre-allocated буферы (нет Allocate в Forward/Backward) ──
    private MemoryBuffer1D<float, Stride1D.Dense> _outputBuffer;
    private MemoryBuffer1D<float, Stride1D.Dense> _gradInputBuffer;

    private readonly Adam _positionOptimizer;
    private bool _disposed;

    // ── Скомпилированные kernels ──
    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _forwardKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        SpecializedValue<int>> _backwardKernel;

    public PositionalEmbeddingLayer(Accelerator accelerator,
        int maxSeqLen, int embeddingDim)
    {
        _accelerator = accelerator;
        MaxSeqLen = maxSeqLen;
        EmbeddingDim = embeddingDim;

        int fullSize = maxSeqLen * embeddingDim;

        PositionWeights = accelerator.Allocate1D<float>(fullSize);
        GradPositionWeights = accelerator.Allocate1D<float>(fullSize);
        _outputBuffer = accelerator.Allocate1D<float>(fullSize);
        _gradInputBuffer = accelerator.Allocate1D<float>(fullSize);

        // Инициализация N(0, 0.02) — как в GPT-2
        var host = new float[fullSize];
        var rnd = new Random(42);
        float std = 0.02f;
        for (int i = 0; i < host.Length; i++)
            host[i] = std * (float)(rnd.NextDouble() * 2 - 1);

        PositionWeights.CopyFromCPU(host);
        GradPositionWeights.MemSetToZero();

        _positionOptimizer = new Adam(accelerator, fullSize);

        // Компилируем kernels один раз
        _forwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(
            PositionalEmbeddingKernels.AddPositionalKernel);

        _backwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            SpecializedValue<int>>(
            PositionalEmbeddingKernels.PosEmbBackwardKernel);
    }

    public string LayerType => "PositionalEmbedding";

    // output[i] = input[i] + posWeights[i],  i in [0, seqLen*EmbeddingDim)
    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        int totalSize = seqLen * EmbeddingDim;
        // Потоков = ceil(totalSize / 4) — каждый обрабатывает 4 элемента
        int threadCount = (totalSize + 3) / 4;

        var outputView = _outputBuffer.View.SubView(0, totalSize);
        var posView = PositionWeights.View.SubView(0, totalSize);

        _forwardKernel(
            threadCount,
            input.SubView(0, totalSize),
            posView,
            outputView,
            SpecializedValue.New(totalSize));

        return outputView;
    }

    // gradInput  = gradOutput          (pass-through)
    // gradPos   += gradOutput[0:seqLen*dim]
    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> gradOutput, float lr)
    {
        int totalSize = (int)gradOutput.IntLength;
        int threadCount = (totalSize + 3) / 4;

        var gradInputView = _gradInputBuffer.View.SubView(0, totalSize);
        var gradPosView = GradPositionWeights.View.SubView(0, totalSize);

        GradPositionWeights.MemSetToZero();

        _backwardKernel(
            threadCount,
            gradOutput.SubView(0, totalSize),
            gradInputView,
            gradPosView,
            SpecializedValue.New(totalSize));

        // ✅ Перегрузка с ArrayView1D — нет ошибки типа
        // activeSize = totalSize → моменты обновляются только для [0..totalSize)
        _positionOptimizer.Step(
            PositionWeights.View.SubView(0, totalSize),
            gradPosView,
            lr,
            totalSize);

        return gradInputView;
    }

    public int Parameters() => MaxSeqLen * EmbeddingDim;

    public void Dispose()
    {
        if (!_disposed)
        {
            PositionWeights.Dispose();
            GradPositionWeights.Dispose();
            _outputBuffer.Dispose();
            _gradInputBuffer.Dispose();
            _positionOptimizer.Dispose();
            _disposed = true;
        }
    }
}