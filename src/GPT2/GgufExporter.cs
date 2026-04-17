//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

namespace LLM.ILGPU;

/// <summary>
/// GGUF Exporter для LLMGPT2 — чистая реализация без Seek/Back-patch.
/// 
/// Структура файла:
///   Header (24 байта) → KV Metadata → Tensor Infos → Padding → Tensor Data
/// 
/// Все смещения вычисляются В ПАМЯТИ до записи файла.
/// Файл записывается за ОДИН последовательный проход.
/// </summary>
public static class GgufExporter
{
    // ═══════════════════════════════════════════════════════════
    // GGUF Constants
    // ═══════════════════════════════════════════════════════════

    private const uint GgufMagic = 0x46554747; // "GGUF" little-endian
    private const uint GgufVersion = 3;
    public const int Alignment = 32;

    // ═══════════════════════════════════════════════════════════
    // Types
    // ═══════════════════════════════════════════════════════════

    private enum GgufType : uint
    {
        UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3,
        UInt32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
        String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12
    }

    private enum GgmlType : uint
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9
    }

    private class Tensor
    {
        public string Name = "";
        public ulong[] Dims = Array.Empty<ulong>();
        public GgmlType Type = GgmlType.F32;
        public ulong Offset;
        public float[] Data = Array.Empty<float>();
    }

    private class KV
    {
        public string Key;
        public GgufType Type;
        public object Value;
    }

    // ═══════════════════════════════════════════════════════════
    // PUBLIC API
    // ═══════════════════════════════════════════════════════════

    public static void Export(
        LLMGPT2 model, ITokenizer tokenizer,
        string outputPath, string modelName = "gpt2")
    {
        Console.WriteLine($"\n═══ ЭКСПОРТ LLMGPT2 В GGUF v3 ═══");

        var layers = model.GetLayers();
        var config = InferConfig(model, layers);

        // ═══ СТРОИМ СТАНДАРТНЫЙ GPT-2 VOCAB С REMAPPING ═══
        var (newTokens, newScores, newTypes, oldToNew) =
            BuildStandardGPT2Vocab(tokenizer);


        // Оригинальные токены по oldId (для верификации)
        var origTokens = tokenizer.Encoder.OrderBy(kv => kv.Value).Select(kv => kv.Key).ToArray();

        var kvList = BuildMetadata(config, tokenizer, modelName, newTokens, newScores, newTypes, oldToNew);
        var tensors = CollectTensors(model, layers, config, oldToNew, newTokens.Length, origTokens);

        // Вычисляем смещения В ПАМЯТИ до записи
        ComputeOffsets(tensors);

        // Записываем файл за ОДИН проход (без Seek!)
        {
            using var fs = File.Open(outputPath, FileMode.Create, FileAccess.Write);
            using var w = new BinaryWriter(fs, Encoding.UTF8);

            // 1. Header (24 байта)
            w.Write(GgufMagic);
            w.Write(GgufVersion);
            w.Write((ulong)tensors.Count);
            w.Write((ulong)kvList.Count);

            // 2. KV Metadata
            foreach (var kv in kvList)
                WriteKV(w, kv);

            // 3. Tensor Info (сразу после KV, без выравнивания)
            foreach (var t in tensors)
                WriteTensorInfo(w, t);

            // 4. Padding до Alignment перед блоком данных
            PadToAlignment(w);

            // 5. Tensor Data (каждый тензор выровнен)
            foreach (var t in tensors)
            {
                WriteTensorData(w, t);
                PadToAlignment(w);
            }

            // Финальный Flush
            w.Flush();
        } // ← здесь файл закрывается

        Console.WriteLine($"\n✅ GGUF создан: {outputPath}");
        Console.WriteLine($"   Размер: {new FileInfo(outputPath).Length / 1024.0 / 1024.0:F2} MB");
    }

    // ═══════════════════════════════════════════════════════════
    // CONFIG
    // ═══════════════════════════════════════════════════════════

    private static ModelConfig InferConfig(LLMGPT2 model, List<ILayer> layers)
    {
        int embDim = 128, hidDim = 256, nHeads = 4, nLayers = 0, maxSeq = 80;

        foreach (var layer in layers)
        {
            switch (layer)
            {
                case EmbeddingLayer emb:
                    embDim = emb.EmbeddingDim;
                    maxSeq = emb.MaxSeqLen;
                    break;
                case TransformerBlockPreNorm block:
                    nLayers++;
                    embDim = block.EmbeddingDim;
                    nHeads = block.NumHeads;
                    hidDim = block.HiddenDim;
                    break;
            }
        }

        return new ModelConfig
        {
            EmbeddingDim = embDim,
            HiddenDim = hidDim,
            NumHeads = nHeads,
            NumLayers = nLayers,
            MaxSeqLen = maxSeq,
            BpeVocabSize = model.Tokenizer.VocabSize,
            // Для нового экспортёра маппим на поля, которые он ожидает
            // VocabSize, ContextLength и т.д. — это те же самые значения
        };
    }

    // ═══════════════════════════════════════════════════════════
    // OFFSET COMPUTATION (в памяти, без записи в файл)
    // ═══════════════════════════════════════════════════════════

    private static void ComputeOffsets(List<Tensor> tensors)
    {
        ulong offset = 0;
        foreach (var t in tensors)
        {
            t.Offset = offset;
            ulong sz = TensorDataBytes(t);
            ulong end = offset + sz;
            ulong pad = ((ulong)Alignment - (end % (ulong)Alignment)) % (ulong)Alignment;
            offset = end + pad;
        }
    }

    private static ulong TensorDataBytes(Tensor t)
    {
        ulong elems = 1;
        foreach (var d in t.Dims) elems *= d;
        return t.Type == GgmlType.F32 ? elems * 4 : elems * 2;
    }

    // ═══════════════════════════════════════════════════════════
    // TENSOR COLLECTION
    // ═══════════════════════════════════════════════════════════

    private static List<Tensor> CollectTensors(
        LLMGPT2 model, List<ILayer> layers, ModelConfig config,
        int[] oldToNew, int newVocabSize, string[] origTokens)
    {
        var tensors = new List<Tensor>();
        int blkIdx = 0;

        int V = config.BpeVocabSize;
        int C = config.EmbeddingDim;
        int T = config.MaxSeqLen;
        int FF = config.HiddenDim;

        foreach (var layer in layers)
        {
            switch (layer)
            {
                case EmbeddingLayer emb:
                    // token_embd.weight: переупорядочиваем строки по new vocab order
                    var embData = GpuToHost(emb.TokenWeights); // [V, C]
                    var newEmb = ReorderEmbeddings(embData, V, C, oldToNew, newVocabSize);
                    tensors.Add(MakeTensor("token_embd.weight",
                        new ulong[] { (ulong)C, (ulong)newVocabSize }, newEmb));
                    break;

                case PositionalEmbeddingLayer pos:
                    // КРИТИЧНО: EmbeddingLayer.Forward() добавляет синусоидальные
                    // позиционные эмбеддинги ЧЕРЕЗ AddPositionalEncodingKernel.
                    // Затем PositionalEmbeddingLayer добавляет обучаемые.
                    // llama.cpp: output = token_embd + position_embd (однократно)
                    // → экспортируем position_embd = learnable + sinusoidal
                    var posData = GpuToHost(pos.PositionWeights);
                    var combinedPos = CombinePositionalEmbeddings(posData, C, T);
                    tensors.Add(MakeTensor("position_embd.weight",
                        new ulong[] { (ulong)C, (ulong)T }, combinedPos));
                    break;

                case TransformerBlockPreNorm block:
                    CollectBlockTensors(block, blkIdx++, tensors, C, FF);
                    break;

                case LayerNormLayer ln when IsFinalNorm(layers, ln):
                    tensors.Add(MakeTensor("output_norm.weight",
                        new ulong[] { (ulong)C }, GpuToHost(ln._gamma)));
                    tensors.Add(MakeTensor("output_norm.bias",
                        new ulong[] { (ulong)C }, GpuToHost(ln._beta)));
                    break;

                case LinearLayer linear when IsLmHead(layers, linear):
                    // output.weight: GGML dims {n_embd, n_vocab}
                    // Переупорядочиваем строки по new vocab order
                    int lmInF = linear._inFeatures;
                    int lmOutF = linear._outFeatures;
                    var lmRaw = GpuToHost(linear._weight);
                    var lmW = TransposeWeight(lmRaw, lmInF, lmOutF); // [outF, inF] = [vocab, embed]
                    var newLmW = ReorderEmbeddings(lmW, lmOutF, lmInF, oldToNew, newVocabSize);
                    tensors.Add(MakeTensor("output.weight",
                        new ulong[] { (ulong)lmInF, (ulong)newVocabSize }, newLmW));
                    break;
            }
        }

        return tensors;
    }

    private static void CollectBlockTensors(
        TransformerBlockPreNorm block, int idx,
        List<Tensor> tensors, int C, int FF)
    {
        string p = $"blk.{idx}";

        // LayerNorm 1 (attn_norm)
        tensors.Add(MakeTensor($"{p}.attn_norm.weight",
            new ulong[] { (ulong)C }, GpuToHost(block._ln1._gamma)));
        tensors.Add(MakeTensor($"{p}.attn_norm.bias",
            new ulong[] { (ulong)C }, GpuToHost(block._ln1._beta)));

        // Fused QKV: сырые данные [inF, outF] каждый → конкатенируем → транспонируем
        int inF = block._attention._wq._inFeatures;
        int outF = block._attention._wq._outFeatures;

        var qW = GpuToHost(block._attention._wq._weight);
        var kW = GpuToHost(block._attention._wk._weight);
        var vW = GpuToHost(block._attention._wv._weight);

        // Конкатенация: для каждой входной строки k: [qW[k, :], kW[k, :], vW[k, :]]
        // Результат: [inF, outF*3] row-major
        var qkvRaw = new float[qW.Length + kW.Length + vW.Length];
        for (int k = 0; k < inF; k++)
        {
            Array.Copy(qW, k * outF, qkvRaw, k * (outF * 3), outF);
            Array.Copy(kW, k * outF, qkvRaw, k * (outF * 3) + outF, outF);
            Array.Copy(vW, k * outF, qkvRaw, k * (outF * 3) + outF * 2, outF);
        }

        // Транспонируем: [inF, outF*3] → [outF*3, inF] row-major для GGML
        var qkvW = TransposeWeight(qkvRaw, inF, outF * 3);

        tensors.Add(MakeTensor($"{p}.attn_qkv.weight",
            new ulong[] { (ulong)inF, (ulong)(outF * 3) }, qkvW));

        // Fused QKV bias
        var qB = block._attention._wq._bias.Length > 0
            ? GpuToHost(block._attention._wq._bias) : new float[outF];
        var kB = block._attention._wk._bias.Length > 0
            ? GpuToHost(block._attention._wk._bias) : new float[outF];
        var vB = block._attention._wv._bias.Length > 0
            ? GpuToHost(block._attention._wv._bias) : new float[outF];

        var qkvB = new float[qB.Length + kB.Length + vB.Length];
        Array.Copy(qB, 0, qkvB, 0, qB.Length);
        Array.Copy(kB, 0, qkvB, qB.Length, kB.Length);
        Array.Copy(vB, 0, qkvB, qB.Length + kB.Length, vB.Length);

        tensors.Add(MakeTensor($"{p}.attn_qkv.bias",
            new ulong[] { (ulong)(outF * 3) }, qkvB));

        // Attention output projection
        AddLinearWithBias(block._attention._wo, $"{p}.attn_output", tensors);

        // LayerNorm 2 (ffn_norm)
        tensors.Add(MakeTensor($"{p}.ffn_norm.weight",
            new ulong[] { (ulong)C }, GpuToHost(block._ln2._gamma)));
        tensors.Add(MakeTensor($"{p}.ffn_norm.bias",
            new ulong[] { (ulong)C }, GpuToHost(block._ln2._beta)));

        // FFN up (c_fc): {inF=embed, outF=hidden}
        AddLinearWithBias(block._ffn.W1, $"{p}.ffn_up", tensors);

        // FFN down (c_proj): {inF=hidden, outF=embed}
        AddLinearWithBias(block._ffn.W2, $"{p}.ffn_down", tensors);
    }

    private static void AddLinearWithBias(LinearLayer linear, string prefix, List<Tensor> tensors)
    {
        int inF = linear._inFeatures;
        int outF = linear._outFeatures;
        // Сырые данные [inF, outF] row-major → транспонируем в [outF, inF] row-major
        // GGML column-major с dims {inF, outF}: offset = i0 + i1*inF = транспонированные данные
        var wRaw = GpuToHost(linear._weight);
        var wT = TransposeWeight(wRaw, inF, outF);

        tensors.Add(MakeTensor($"{prefix}.weight",
            new ulong[] { (ulong)inF, (ulong)outF }, wT));

        float[] bias;
        if (linear._bias.Length > 0)
            bias = GpuToHost(linear._bias);
        else
            bias = new float[outF];

        tensors.Add(MakeTensor($"{prefix}.bias",
            new ulong[] { (ulong)outF }, bias));
    }

    private static bool IsFinalNorm(List<ILayer> layers, LayerNormLayer ln)
    {
        for (int i = layers.Count - 1; i >= 0; i--)
            if (layers[i] is LayerNormLayer) return layers[i] == ln;
        return false;
    }

    private static bool IsLmHead(List<ILayer> layers, LinearLayer linear)
    {
        for (int i = layers.Count - 1; i >= 0; i--)
            if (layers[i] is LinearLayer) return layers[i] == linear;
        return false;
    }

    private static Tensor MakeTensor(string name, ulong[] dims, float[] data) =>
        new Tensor { Name = name, Dims = dims, Type = GgmlType.F32, Data = data };

    private static float[] GpuToHost(MemoryBuffer1D<float, Stride1D.Dense> buf)
    {
        var host = new float[buf.Length];
        buf.CopyToCPU(host);
        return host;
    }

    private static float[] TransposeWeight(float[] data, int inF, int outF)
    {
        var result = new float[data.Length];
        for (int k = 0; k < inF; k++)
            for (int j = 0; j < outF; j++)
                result[j * inF + k] = data[k * outF + j];
        return result;
    }

    /// <summary>
    /// Комбинирует обучаемые и синусоидальные позиционные эмбеддинги.
    /// 
    /// EmbeddingLayer.Forward() добавляет синусоидальные PE через AddPositionalEncodingKernel:
    ///   divTerm = 10000^(2*(d/2) / embeddingDim)   // d/2 — целочисленное деление
    ///   angle = pos / divTerm
    ///   pe = (d%2==0) ? Sin(angle) : Cos(angle)
    /// 
    /// Затем PositionalEmbeddingLayer добавляет обучаемые PE.
    /// llama.cpp делает token_embd + position_embd однократно,
    /// поэтому экспортируем position_embd = learnable + sinusoidal.
    /// </summary>
    private static float[] CombinePositionalEmbeddings(float[] learnable, int embDim, int maxSeq)
    {
        var combined = new float[learnable.Length];
        Array.Copy(learnable, combined, learnable.Length);

        for (int pos = 0; pos < maxSeq; pos++)
        {
            for (int d = 0; d < embDim; d++)
            {
                // Та же формула что в AddPositionalEncodingKernel
                int halfD = d / 2; // целочисленное деление
                float divTerm = (float)Math.Pow(10000.0, (2.0 * halfD) / embDim);
                float angle = pos / divTerm;
                float peValue = (d % 2 == 0) ? (float)Math.Sin(angle) : (float)Math.Cos(angle);

                combined[pos * embDim + d] += peValue;
            }
        }

        return combined;
    }

    // ═══════════════════════════════════════════════════════════
    // METADATA
    // ═══════════════════════════════════════════════════════════

    private static List<KV> BuildMetadata(
        ModelConfig config, ITokenizer tokenizer, string modelName,
        string[] tokens, float[] scores, int[] types, int[] oldToNew)
    {
        const string arch = "gpt2";
        var kvs = new List<KV>();

        // General
        kvs.Add(MakeKV("general.architecture", GgufType.String, arch));
        kvs.Add(MakeKV("general.name", GgufType.String, modelName));
        kvs.Add(MakeKV("general.type", GgufType.String, "model"));
        kvs.Add(MakeKV("general.file_type", GgufType.UInt32, 0u)); // F32 ALL_F32 полная точность
        kvs.Add(MakeKV("general.alignment", GgufType.UInt32, (uint)Alignment));

        // Architecture (GPT-2 specific)
        kvs.Add(MakeKV($"{arch}.context_length", GgufType.UInt32, (uint)config.MaxSeqLen));
        kvs.Add(MakeKV($"{arch}.embedding_length", GgufType.UInt32, (uint)config.EmbeddingDim));
        kvs.Add(MakeKV($"{arch}.feed_forward_length", GgufType.UInt32, (uint)config.HiddenDim));
        kvs.Add(MakeKV($"{arch}.block_count", GgufType.UInt32, (uint)config.NumLayers));
        kvs.Add(MakeKV($"{arch}.attention.head_count", GgufType.UInt32, (uint)config.NumHeads));
        // GPT-2 не использует GQA — head_count_kv НЕ записываем
        kvs.Add(MakeKV($"{arch}.attention.layer_norm_epsilon", GgufType.Float32, 1e-5f));
        kvs.Add(MakeKV($"{arch}.vocab_size", GgufType.UInt32, (uint)tokens.Length));

        // Tokenizer — GPT-2 byte-level
        kvs.Add(MakeKV("tokenizer.ggml.model", GgufType.String, "gpt2"));

        string[] merges = tokenizer.Merges.Select(m => $"{m.A} {m.B}").ToArray();

        kvs.Add(MakeArrayKV("tokenizer.ggml.tokens", tokens));
        kvs.Add(MakeArrayKV("tokenizer.ggml.scores", scores));
        kvs.Add(MakeArrayKV("tokenizer.ggml.token_type", types));
        kvs.Add(MakeArrayKV("tokenizer.ggml.merges", merges));

        // Special token ids — пересчитываем через remapping:
        int RemapSpecialToken(int oldId)
        {
            if (oldId >= 0 && oldId < oldToNew.Length)
                return oldToNew[oldId];
            return -1;
        }

        int newBosId = RemapSpecialToken(tokenizer.BosId);
        int newEosId = RemapSpecialToken(tokenizer.EosId);
        int newUnkId = RemapSpecialToken(tokenizer.UnkId);
        int newPadId = RemapSpecialToken(tokenizer.PadId);

        // Если remapping не сработал, ищем по имени в новом vocab:
        if (newBosId < 0) newBosId = Array.IndexOf(tokens, "<s>");
        if (newEosId < 0) newEosId = Array.IndexOf(tokens, "</s>");
        if (newUnkId < 0) newUnkId = Array.IndexOf(tokens, "<unk>");
        if (newPadId < 0) newPadId = Array.IndexOf(tokens, "<pad>");

        // Записываем ID спецтокенов (если не найдены — используем 0)
        uint uBos = newBosId >= 0 ? (uint)newBosId : 0u;
        uint uEos = newEosId >= 0 ? (uint)newEosId : 0u;
        uint uUnk = newUnkId >= 0 ? (uint)newUnkId : 0u;
        uint uPad = newPadId >= 0 ? (uint)newPadId : 0u;

        kvs.Add(MakeKV("tokenizer.ggml.bos_token_id", GgufType.UInt32, uBos));
        kvs.Add(MakeKV("tokenizer.ggml.eos_token_id", GgufType.UInt32, uEos));
        kvs.Add(MakeKV("tokenizer.ggml.unknown_token_id", GgufType.UInt32, uUnk));
        kvs.Add(MakeKV("tokenizer.ggml.padding_token_id", GgufType.UInt32, uPad));

        kvs.Add(MakeKV("tokenizer.ggml.add_space_prefix", GgufType.Bool, false));

        // БЕЗ автоматического BOS/EOS — промпт строится вручную по формату обучения
        kvs.Add(MakeKV("tokenizer.ggml.add_bos_token", GgufType.Bool, true));
        kvs.Add(MakeKV("tokenizer.ggml.add_eos_token", GgufType.Bool, false));

        // Chat template — ТОЧНО воспроизводит формат обучения:
        // '<user> {content} <assistant> {response} </s>'
        //чат с историей, не используем
        string chatTemplateHistory =
            "{% for message in messages %}" +
            "{% if message['role'] == 'user' %}" +
            "<user> {{ message['content'] }}<assistant>" +   // ← БЕЗ пробела перед <assistant>
            "{% elif message['role'] == 'assistant' %}" +
            " {{ message['content'] }}</s>" +
            "{% endif %}" +
            "{% endfor %}";

        //чат без истории
        string chatTemplate =
            "{% if messages %}" +
            "{% set message = messages[-1] %}" +          // только последнее сообщение
            "{% if message['role'] == 'user' %}" +
            "<user> {{ message['content'] }}<assistant>" +
            "{% endif %}" +
            "{% endif %}";

        kvs.Add(MakeKV("tokenizer.chat_template", GgufType.String, chatTemplate));

        return kvs;
    }

    // ═══════════════════════════════════════════════════════════
    // FILE WRITING
    // ═══════════════════════════════════════════════════════════

    private static void SkipValue(BinaryReader r, GgufType type)
    {
        switch (type)
        {
            case GgufType.UInt8: case GgufType.Int8: r.ReadByte(); break;
            case GgufType.UInt16: case GgufType.Int16: r.ReadUInt16(); break;
            case GgufType.UInt32: case GgufType.Int32: r.ReadUInt32(); break;
            case GgufType.Float32: r.ReadSingle(); break;
            case GgufType.Bool: r.ReadByte(); break;
            case GgufType.String:
                ulong len = r.ReadUInt64();
                r.BaseStream.Seek((long)len, SeekOrigin.Current);
                break;
            case GgufType.UInt64: case GgufType.Int64: r.ReadUInt64(); break;
            case GgufType.Float64: r.ReadDouble(); break;
            case GgufType.Array:
                var elemType = (GgufType)r.ReadUInt32();
                ulong count = r.ReadUInt64();
                for (ulong j = 0; j < count; j++)
                    SkipValue(r, elemType);
                break;
        }
    }

    private static void WriteKV(BinaryWriter w, KV kv)
    {
        WriteString(w, kv.Key);
        w.Write((uint)kv.Type);

        if (kv.Type == GgufType.Array)
        {
            var arr = (GgufArray)kv.Value!;
            w.Write((uint)arr.ElemType);
            w.Write((ulong)arr.Items.Count);
            foreach (var item in arr.Items.Cast<object>())
                WriteGgufValue(w, arr.ElemType, item);
        }
        else
        {
            WriteGgufValue(w, kv.Type, kv.Value!);
        }
    }

    private static void WriteGgufValue(BinaryWriter w, GgufType type, object val)
    {
        switch (type)
        {
            case GgufType.UInt8: w.Write((byte)val); break;
            case GgufType.Int8: w.Write((sbyte)val); break;
            case GgufType.UInt16: w.Write((ushort)val); break;
            case GgufType.Int16: w.Write((short)val); break;
            case GgufType.UInt32: w.Write((uint)val); break;
            case GgufType.Int32: w.Write((int)val); break;
            case GgufType.Float32: w.Write((float)val); break;
            case GgufType.Bool: w.Write((byte)((bool)val ? 1 : 0)); break;
            case GgufType.String: WriteString(w, (string)val); break;
            case GgufType.UInt64: w.Write((ulong)val); break;
            case GgufType.Int64: w.Write((long)val); break;
            case GgufType.Float64: w.Write((double)val); break;
            default: throw new NotSupportedException($"Unsupported GGUF type: {type}");
        }
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void WriteTensorInfo(BinaryWriter w, Tensor t)
    {
        WriteString(w, t.Name);
        w.Write((uint)t.Dims.Length);
        foreach (var d in t.Dims) w.Write(d);
        w.Write((uint)t.Type);
        w.Write(t.Offset);
    }

    private static void WriteTensorData(BinaryWriter w, Tensor t)
    {
        var bytes = new byte[t.Data.Length * sizeof(float)];
        Buffer.BlockCopy(t.Data, 0, bytes, 0, bytes.Length);
        w.Write(bytes);
    }

    private static void PadToAlignment(BinaryWriter w)
    {
        long pos = w.BaseStream.Position;
        long pad = (Alignment - (pos % Alignment)) % Alignment;
        for (long i = 0; i < pad; i++) w.Write((byte)0);
    }

    // ═══════════════════════════════════════════════════════════
    // KV HELPERS
    // ═══════════════════════════════════════════════════════════

    private static KV MakeKV(string key, GgufType type, object value) =>
        new KV { Key = key, Type = type, Value = value };

    // ═══════════════════════════════════════════════════════════
    // GPT-2 BYTE-LEVEL VOCAB REORDERING
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Переупорядочивает строки матрицы эмбеддингов по remapping oldId → newId.
    /// data: [numRows, rowWidth] row-major. newId-я строка = oldId-я строка исходных данных.
    /// </summary>
    private static float[] ReorderEmbeddings(float[] data, int numRows, int rowWidth,
        int[] oldToNew, int newNumRows)
    {
        var result = new float[newNumRows * rowWidth];
        for (int oldId = 0; oldId < numRows; oldId++)
        {
            int newId = oldToNew[oldId];
            if (newId < newNumRows)
                Array.Copy(data, oldId * rowWidth, result, newId * rowWidth, rowWidth);
        }
        return result;
    }

    /// <summary>
    /// Строит стандартный GPT-2 vocab с remapping: oldId → newId.
    /// Специальные токены остаются на id 0-6.
    /// Byte-токены идут в стандартном GPT-2 порядке (пропуская специальные).
    /// Merge-токены сохраняют относительный порядок.
    /// </summary>
    private static (string[] tokens, float[] scores, int[] types, int[] oldToNew)
        BuildStandardGPT2Vocab(ITokenizer tokenizer)
    {
        var oldTokens = tokenizer.Encoder
            .OrderBy(kv => kv.Value)
            .Select(kv => kv.Key)
            .ToArray();
        int V = oldTokens.Length;

        // ═══ Стандартная таблица GPT-2 bytes_to_unicode() ═══
        var gpt2 = new char[256];
        {
            var bs = new List<int>();
            for (int i = 33;  i <= 126; i++) bs.Add(i);
            for (int i = 161; i <= 172; i++) bs.Add(i);
            for (int i = 174; i <= 255; i++) bs.Add(i);
            var cs = new List<int>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
                if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n++); }
            for (int i = 0; i < bs.Count; i++)
                gpt2[bs[i]] = (char)cs[i];
        }

        // Стандартный порядок 256 байт:
        var byteOrder = new List<int>();
        for (int i = 33;  i <= 126; i++) byteOrder.Add(i);
        for (int i = 161; i <= 172; i++) byteOrder.Add(i);
        for (int i = 174; i <= 255; i++) byteOrder.Add(i);
        for (int b = 0; b < 256; b++)
            if (!byteOrder.Contains(b)) byteOrder.Add(b);

        // ═══ Специальные токены (фиксированные id 0..6) ═══
        var specials = new[]
        { "<pad>", "<unk>", "<s>", "</s>", "<user>", "<assistant>", "<sep>" };
        var specialSet = new HashSet<string>(specials);

        // ═══ Строим новый vocab ═══
        var newVocab = new List<string>(specials); // id 0..6
        var inVocab  = new HashSet<string>(specials);

        // Byte-токены (пропускаем если совпадают со специальными):
        int byteCount = 0;
        foreach (int b in byteOrder)
        {
            string tok = gpt2[b].ToString();
            if (!inVocab.Contains(tok))
            {
                newVocab.Add(tok);
                inVocab.Add(tok);
                byteCount++;
            }
        }

        // Merge-токены (всё остальное из старого vocab):
        int mergeCount = 0;
        foreach (string tok in oldTokens)
        {
            if (!inVocab.Contains(tok))
            {
                newVocab.Add(tok);
                inVocab.Add(tok);
                mergeCount++;
            }
        }

        Console.WriteLine($"  Специальных:  {specials.Length}");
        Console.WriteLine($"  Byte-токенов: {byteCount}");
        Console.WriteLine($"  Merge-токенов:{mergeCount}");
        Console.WriteLine($"  Итого:        {newVocab.Count}  (было {V})");

        // ═══ oldId → newId ═══
        var tokToNewId = newVocab
            .Select((t, i) => (t, i))
            .ToDictionary(x => x.t, x => x.i);

        int[] oldToNew = oldTokens
            .Select(t => tokToNewId.TryGetValue(t, out int nid) ? nid : -1)
            .ToArray();

        // ═══ Scores ═══
        var rank = new Dictionary<string, int>();
        for (int i = 0; i < tokenizer.Merges.Count; i++)
        {
            string r = tokenizer.Merges[i].A + tokenizer.Merges[i].B;
            if (!rank.ContainsKey(r)) rank[r] = i;
        }

        float[] scores = newVocab.Select(tok => tok switch {
            "<pad>" => 0f,
            "<unk>" => 0f,
            "<s>" => 0f,
            "</s>" => 0f,
            "<sep>" => 0f,
            "<user>" => 0f,
            "<assistant>" => 0f,
            _ => rank.TryGetValue(tok, out int r) ? -(float)r : -1f
        }).ToArray();

        var ctrl = new HashSet<string> { "<pad>", "<unk>", "<s>", "</s>", "<sep>", "<user>", "<assistant>" };
        int[] types = newVocab.Select(tok => ctrl.Contains(tok) ? 3 : 1).ToArray();

        return (newVocab.ToArray(), scores, types, oldToNew);
    }

    private static KV MakeArrayKV(string key, string[] items) =>
        new KV { Key = key, Type = GgufType.Array, Value = new GgufArray(GgufType.String, items.Cast<object>().ToList()) };

    private static KV MakeArrayKV(string key, float[] items) =>
        new KV { Key = key, Type = GgufType.Array, Value = new GgufArray(GgufType.Float32, items.Cast<object>().ToList()) };

    private static KV MakeArrayKV(string key, int[] items) =>
        new KV { Key = key, Type = GgufType.Array, Value = new GgufArray(GgufType.Int32, items.Cast<object>().ToList()) };

    private class GgufArray
    {
        public GgufType ElemType;
        public System.Collections.Generic.IReadOnlyList<object> Items;
        public GgufArray(GgufType elemType, System.Collections.Generic.IReadOnlyList<object> items)
        { ElemType = elemType; Items = items; }
    }
}
