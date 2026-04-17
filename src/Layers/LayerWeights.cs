//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using System.Reflection;

namespace LLM.ILGPU;

/// <summary>
/// Оптимизированный класс для сохранения и загрузки весов GPU слоёв
/// С безопасной работой с Reflection и быстрой сериализацией через Buffer.BlockCopy
/// </summary>
public static class LayerWeights
{
    private const BindingFlags Flags =
        BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    /// <summary>
    /// Вспомогательный метод для безопасного получения значения поля
    /// Ищет поле в текущем классе и в базовых классах
    /// </summary>
    private static object GetFieldValue(object obj, string fieldName)
    {
        var type = obj.GetType();
        var field = type.GetField(fieldName, Flags);

        if (field == null && type.BaseType != null)
            field = type.BaseType.GetField(fieldName, Flags);

        if (field == null)
            throw new InvalidOperationException(
                $"Поле '{fieldName}' не найдено в типе {type.Name}");

        var value = field.GetValue(obj);
        if (value == null)
            throw new NullReferenceException(
                $"Поле '{fieldName}' в {type.Name} содержит null");

        return value;
    }

    // ─── Buffer save/load ────────────────────────────────────────

    /// <summary>
    /// Оптимизированное сохранение буфера (быстрая запись массива через Buffer.BlockCopy)
    /// </summary>
    public static void SaveBuffer(MemoryBuffer1D<float, Stride1D.Dense>? buffer,
        BinaryWriter writer, string name = "")
    {
        if (buffer == null)
        {
            writer.Write(0);
            return;
        }
        int length = (int)buffer.Length;
        writer.Write(length);
        float[] host = new float[length];
        buffer.View.CopyToCPU(host);
        byte[] byteArray = new byte[length * sizeof(float)];
        Buffer.BlockCopy(host, 0, byteArray, 0, byteArray.Length);
        writer.Write(byteArray);
    }

    /// <summary>
    /// Загрузка буфера с оптимизированным чтением через Buffer.BlockCopy
    /// </summary>
    public static void LoadBuffer(MemoryBuffer1D<float, Stride1D.Dense>? buffer,
        BinaryReader reader, string name = "")
    {
        int savedLength = reader.ReadInt32();
        if (savedLength == 0) return;

        if (buffer == null || buffer.Length != savedLength)
        {
            reader.BaseStream.Seek(savedLength * sizeof(float),
                SeekOrigin.Current);
            return;
        }

        byte[] byteArray = reader.ReadBytes(savedLength * sizeof(float));
        float[] host = new float[savedLength];
        Buffer.BlockCopy(byteArray, 0, host, 0, byteArray.Length);
        buffer.View.CopyFromCPU(host);
    }

    // ─── Linear ──────────────────────────────────────────────────

    /// <summary>
    /// Сохранить веса Linear слоя
    /// </summary>
    public static void SaveLinearWeights(object layer,
        BinaryWriter writer, string layerName)
    {
        var weight = GetFieldValue(layer, "_weight")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        var bias = GetFieldValue(layer, "_bias")
            as MemoryBuffer1D<float, Stride1D.Dense>;

        writer.Write($"Linear:{layerName}");
        SaveBuffer(weight, writer, $"{layerName}.weight");
        SaveBuffer(bias, writer, $"{layerName}.bias");
    }

    /// <summary>
    /// Загрузить веса Linear слоя
    /// </summary>
    public static void LoadLinearWeights(object layer,
        BinaryReader reader, string layerName)
    {
        var marker = reader.ReadString();
        if (marker != $"Linear:{layerName}")
            throw new InvalidDataException(
                $"Ошибка весов! Ждали Linear:{layerName}, получили {marker}");

        var weight = GetFieldValue(layer, "_weight")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        var bias = GetFieldValue(layer, "_bias")
            as MemoryBuffer1D<float, Stride1D.Dense>;

        LoadBuffer(weight, reader, $"{layerName}.weight");
        LoadBuffer(bias, reader, $"{layerName}.bias");
    }

    // ─── LayerNorm ───────────────────────────────────────────────

    /// <summary>
    /// Сохранить веса LayerNorm слоя
    /// </summary>
    public static void SaveLayerNormWeights(object layer,
        BinaryWriter writer, string layerName)
    {
        var gamma = GetFieldValue(layer, "_gamma")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        var beta = GetFieldValue(layer, "_beta")
            as MemoryBuffer1D<float, Stride1D.Dense>;

        writer.Write($"LayerNorm:{layerName}");
        SaveBuffer(gamma, writer, $"{layerName}.gamma");
        SaveBuffer(beta, writer, $"{layerName}.beta");
    }

    /// <summary>
    /// Загрузить веса LayerNorm слоя
    /// </summary>
    public static void LoadLayerNormWeights(object layer,
        BinaryReader reader, string layerName)
    {
        var marker = reader.ReadString();
        if (marker != $"LayerNorm:{layerName}")
            throw new InvalidDataException(
                $"Ошибка весов! Ждали LayerNorm:{layerName}, получили {marker}");

        var gamma = GetFieldValue(layer, "_gamma")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        var beta = GetFieldValue(layer, "_beta")
            as MemoryBuffer1D<float, Stride1D.Dense>;

        LoadBuffer(gamma, reader, $"{layerName}.gamma");
        LoadBuffer(beta, reader, $"{layerName}.beta");
    }

    // ─── Embedding ───────────────────────────────────────────────

    /// <summary>
    /// Сохранить веса EmbeddingLayerGPU
    /// </summary>
    public static void SaveEmbeddingGPUWeights(EmbeddingLayer layer,
        BinaryWriter writer)
    {
        writer.Write("Embedding");
        var tokenWeights = GetFieldValue(layer, "TokenWeights")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        SaveBuffer(tokenWeights, writer, "tokenWeights");
    }

    /// <summary>
    /// Загрузить веса EmbeddingLayerGPU
    /// </summary>
    public static void LoadEmbeddingGPUWeights(EmbeddingLayer layer,
        BinaryReader reader)
    {
        var marker = reader.ReadString();
        if (marker != "Embedding")
            throw new InvalidDataException(
                $"Ошибка маркера! Ждали 'EmbeddingGPU', получили {marker}");

        var tokenWeights = GetFieldValue(layer, "TokenWeights")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        LoadBuffer(tokenWeights, reader, "tokenWeights");
    }

    // ─── PositionalEmbedding ────────────────────────────────────

    public static void SavePositionalEmbeddingGPUWeights(
        PositionalEmbeddingLayer layer, BinaryWriter writer)
    {
        writer.Write("PositionalEmbedding");
        var posWeights = GetFieldValue(layer, "PositionWeights")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        SaveBuffer(posWeights, writer, "positionWeights");
    }

    public static void LoadPositionalEmbeddingGPUWeights(
        PositionalEmbeddingLayer layer, BinaryReader reader)
    {
        var marker = reader.ReadString();
        if (marker != "PositionalEmbedding")
            throw new InvalidDataException(
                $"Ошибка маркера! Ждали 'PositionalEmbedding', получили {marker}");

        var posWeights = GetFieldValue(layer, "PositionWeights")
            as MemoryBuffer1D<float, Stride1D.Dense>;
        LoadBuffer(posWeights, reader, "positionWeights");
    }

    public static void SavePreNormBlockWeights(
        TransformerBlockPreNorm block, BinaryWriter writer)
    {
        // LN1 (attn_norm)
        LayerWeights.SaveLayerNormWeights(block._ln1, writer, "ln1");

        // Attention
        LayerWeights.SaveLinearWeights(block._attention._wq, writer, "wq");
        LayerWeights.SaveLinearWeights(block._attention._wk, writer, "wk");
        LayerWeights.SaveLinearWeights(block._attention._wv, writer, "wv");
        LayerWeights.SaveLinearWeights(block._attention._wo, writer, "wo");

        // LN2 (ffn_norm)
        LayerWeights.SaveLayerNormWeights(block._ln2, writer, "ln2");

        // FFN (GELU)
        LayerWeights.SaveLinearWeights(block._ffn.W1, writer, "w1");
        LayerWeights.SaveLinearWeights(block._ffn.W2, writer, "w2");
    }

    public static void LoadPreNormBlockWeights(
        TransformerBlockPreNorm block, BinaryReader reader)
    {
        LayerWeights.LoadLayerNormWeights(block._ln1, reader, "ln1");

        LayerWeights.LoadLinearWeights(block._attention._wq, reader, "wq");
        LayerWeights.LoadLinearWeights(block._attention._wk, reader, "wk");
        LayerWeights.LoadLinearWeights(block._attention._wv, reader, "wv");
        LayerWeights.LoadLinearWeights(block._attention._wo, reader, "wo");

        LayerWeights.LoadLayerNormWeights(block._ln2, reader, "ln2");

        LayerWeights.LoadLinearWeights(block._ffn.W1, reader, "w1");
        LayerWeights.LoadLinearWeights(block._ffn.W2, reader, "w2");
    }

    /// <summary>
    /// Получает список слоёв из LLMGPT2 через reflection.
    /// </summary>
    public static List<ILayer> GetNetworkLayers(ILLM llm)
    {
        if (llm is LLMGPT2 gpt2)
            return gpt2.GetLayers();

        throw new NotSupportedException(
            $"GetNetworkLayers не поддерживает тип {llm.GetType().Name}");
    }

    /// <summary>
    /// Reflection helper для доступа к private полям.
    /// </summary>
    public static T GetField<T>(object obj, string fieldName) where T : class
    {
        var type = obj.GetType();
        var field = type.GetField(fieldName,
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Public);

        if (field == null && type.BaseType != null)
            field = type.BaseType.GetField(fieldName,
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Public);

        return (T)(field?.GetValue(obj)
            ?? throw new InvalidOperationException(
                $"Поле {fieldName} не найдено в {type.Name}"));
    }
}
