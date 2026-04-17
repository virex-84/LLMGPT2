//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using System;

namespace LLM.ILGPU;

/// <summary>
/// Обучаемые позиционные эмбеддинги (GPT-2 style).
/// В отличие от синусоидальных — это обучаемые параметры для каждой позиции.
/// </summary>
public class PositionalEmbeddingLayer : ILayer, IDisposable
{
    private readonly Accelerator _accelerator;
    public int MaxSeqLen { get; }
    public int EmbeddingDim { get; }

    public MemoryBuffer1D<float, Stride1D.Dense> PositionWeights;
    public MemoryBuffer1D<float, Stride1D.Dense> GradPositionWeights;

    private readonly Adam _positionOptimizer;
    private bool _disposed;

    public PositionalEmbeddingLayer(Accelerator accelerator, int maxSeqLen, int embeddingDim)
    {
        _accelerator = accelerator;
        MaxSeqLen = maxSeqLen;
        EmbeddingDim = embeddingDim;

        PositionWeights = accelerator.Allocate1D<float>(maxSeqLen * embeddingDim);
        GradPositionWeights = accelerator.Allocate1D<float>(maxSeqLen * embeddingDim);

        // Инициализация N(0, 0.02) — как в GPT-2
        float std = 0.02f;
        var host = new float[maxSeqLen * embeddingDim];
        var rnd = new Random(42);
        for (int i = 0; i < host.Length; i++)
            host[i] = std * (float)(rnd.NextDouble() * 2 - 1);
        PositionWeights.CopyFromCPU(host);
        GradPositionWeights.MemSetToZero();

        _positionOptimizer = new Adam(accelerator, (int)PositionWeights.Length);
    }

    public string LayerType => "PositionalEmbedding";

    /// <summary>
    /// Добавляет позиционные эмбеддинги к входным данным.
    /// output = input + posEmb[0:seqLen, :]
    /// </summary>
    public ArrayView1D<float, Stride1D.Dense> Forward(
        ArrayView1D<float, Stride1D.Dense> input, int seqLen)
    {
        int totalSize = seqLen * EmbeddingDim;

        // Копируем данные на CPU
        var inputHost = new float[totalSize];
        input.SubView(0, totalSize).CopyToCPU(inputHost);

        var posFull = new float[MaxSeqLen * EmbeddingDim];
        PositionWeights.CopyToCPU(posFull);

        // output = input + posEmb[0:seqLen, :]
        var outputHost = new float[totalSize];
        for (int i = 0; i < totalSize; i++)
            outputHost[i] = inputHost[i] + posFull[i];

        var output = _accelerator.Allocate1D<float>(totalSize);
        output.CopyFromCPU(outputHost);
        return output.View;
    }

    public ArrayView1D<float, Stride1D.Dense> Backward(
        ArrayView1D<float, Stride1D.Dense> gradOutput, float lr)
    {
        int totalSize = (int)gradOutput.IntLength;
        var gradInput = _accelerator.Allocate1D<float>(totalSize);

        // Градиент по входу — копия gradOutput
        gradInput.View.SubView(0, totalSize).CopyFrom(gradOutput);

        // Градиент по позиционным весам — аккумулируем
        int seqLen = totalSize / EmbeddingDim;

        // Копируем на CPU
        var gradHost = new float[totalSize];
        gradOutput.SubView(0, totalSize).CopyToCPU(gradHost);

        int fullSize = MaxSeqLen * EmbeddingDim;
        var gradPosHost = new float[fullSize];
        GradPositionWeights.CopyToCPU(gradPosHost);

        // Аккумулируем градиенты только для использованных позиций
        for (int i = 0; i < seqLen; i++)
            for (int d = 0; d < EmbeddingDim; d++)
            {
                int idx = i * EmbeddingDim + d;
                gradPosHost[idx] += gradHost[idx];
            }

        GradPositionWeights.CopyFromCPU(gradPosHost);
        _positionOptimizer.Step(PositionWeights, GradPositionWeights, lr);

        return gradInput.View;
    }

    public int Parameters() => MaxSeqLen * EmbeddingDim;

    public void Dispose()
    {
        if (!_disposed)
        {
            PositionWeights.Dispose();
            GradPositionWeights.Dispose();
            _positionOptimizer.Dispose();
            _disposed = true;
        }
    }
}
