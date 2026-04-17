//https://github.com/virex-84

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLM.ILGPU;

/// <summary>
/// BpeTokenizer для корректного экспорта в GGUF/llama.cpp.
/// - Спецтокены принудительно добавляются в vocab ДО BPE обучения
/// - Спецтокены регистрируются как атомарные merges
/// - PreTokenize использует GPT-2 regex с учётом спецтокенов
/// - Формат диалога строго "<user> ... <assistant> ... </s>"
/// </summary>
public class BPETokenizer : ITokenizer
{
    // ═══════════════════════════════════════════════════════
    // GPT-2 byte↔unicode mapping
    // ═══════════════════════════════════════════════════════
    public static readonly Dictionary<byte, char> ByteToUnicode;
    public static readonly Dictionary<char, byte> UnicodeToByte;

    static BPETokenizer()
    {
        ByteToUnicode = BuildBytesToUnicode();
        UnicodeToByte = ByteToUnicode.ToDictionary(kv => kv.Value, kv => kv.Key);
    }

    private static Dictionary<byte, char> BuildBytesToUnicode()
    {
        // Точная копия GPT-2 bytes_to_unicode()
        var bs = Enumerable.Range('!', '~' - '!' + 1)
            .Concat(Enumerable.Range('\u00A1', '\u00AC' - '\u00A1' + 1))
            .Concat(Enumerable.Range('\u00AE', '\u00FF' - '\u00AE' + 1))
            .Select(i => (byte)i).ToList();
        var cs = bs.Select(b => (int)b).ToList();
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!bs.Contains((byte)b)) { bs.Add((byte)b); cs.Add(256 + n++); }
        var d = new Dictionary<byte, char>();
        for (int i = 0; i < bs.Count; i++) d[bs[i]] = (char)cs[i];
        return d;
    }

    // ═══════════════════════════════════════════════════════
    // Специальные токены — СТРОГО как в chat_template GGUF
    // ═══════════════════════════════════════════════════════
    public const string PadToken = "<pad>";
    public const string UnkToken = "<unk>";
    public const string BosToken = "<s>";
    public const string EosToken = "</s>";
    public const string UserToken = "<user>";
    public const string AssistantToken = "<assistant>";
    public const string SepToken = "<sep>";

    // ПОРЯДОК ВАЖЕН: совпадает с BuildStandardGPT2Vocab в экспортере
    private static readonly string[] SpecialTokensOrder = new[]
    {
        PadToken, UnkToken, BosToken, EosToken,
        UserToken, AssistantToken, SepToken
    };

    // ═══════════════════════════════════════════════════════
    // Состояние токенизатора
    // ═══════════════════════════════════════════════════════
    public Dictionary<string, int> Vocab { get; private set; } = new();
    public List<(string A, string B)> Merges { get; private set; } = new();
    public List<string> SpecialTokens { get; private set; }

    private Dictionary<(string, string), int>? _ranks;
    private Dictionary<string, string[]> _bpeCache = new();

    // ═══════════════════════════════════════════════════════
    // ITokenizer implementation
    // ═══════════════════════════════════════════════════════
    IReadOnlyDictionary<string, int> ITokenizer.Encoder => Vocab;
    IReadOnlyList<(string A, string B)> ITokenizer.Merges => Merges;

    public int VocabSize => Vocab.Count;
    public int PadId => Vocab.TryGetValue(PadToken, out var i) ? i : 0;
    public int UnkId => Vocab.TryGetValue(UnkToken, out var i) ? i : 1;
    public int BosId => Vocab.TryGetValue(BosToken, out var i) ? i : 2;
    public int EosId => Vocab.TryGetValue(EosToken, out var i) ? i : 3;
    public int UserId => Vocab.TryGetValue(UserToken, out var i) ? i : 4;
    public int AssistantId => Vocab.TryGetValue(AssistantToken, out var i) ? i : 5;

    public BPETokenizer(IEnumerable<string>? specialTokens = null)
    {
        SpecialTokens = specialTokens?.ToList() ?? SpecialTokensOrder.ToList();
    }

    // ═══════════════════════════════════════════════════════
    // GPT-2 Pre-tokenization regex
    // КРИТИЧНО: спецтокены должны быть в regex ПЕРВЫМИ
    // ═══════════════════════════════════════════════════════
    private Regex? _preTokenizeRegex;

    private Regex GetPreTokenizeRegex()
    {
        if (_preTokenizeRegex != null) return _preTokenizeRegex;

        // Экранируем спецтокены — матчим с опциональным пробелом перед ними
        var escapedSpecials = SpecialTokens
            .OrderByDescending(s => s.Length)
            .Select(s => @" ?" + Regex.Escape(s));  // ← добавить " ?"

        string specialsPart = string.Join("|", escapedSpecials);

        string pattern =
            specialsPart + "|" +
            @"'s|'t|'re|'ve|'m|'ll|'d" +
            @"| ?\p{L}+" +
            @"| ?\p{N}+" +
            @"| ?[^\s\p{L}\p{N}]+" +
            @"|\s+(?!\S)" +
            @"|\s+";

        _preTokenizeRegex = new Regex(pattern,
            RegexOptions.Compiled | RegexOptions.CultureInvariant);
        return _preTokenizeRegex;
    }

    // ═══════════════════════════════════════════════════════
    // ОБУЧЕНИЕ
    // ═══════════════════════════════════════════════════════

    /// <summary>
    /// Обучает BPE с гарантией корректного экспорта в GGUF.
    /// </summary>
    public void Train(
        IEnumerable<string> corpus,
        int vocabSize = 1024,
        int minFrequency = 1,
        bool verbose = true)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Vocab = new Dictionary<string, int>();
        Merges = new List<(string, string)>();
        _ranks = null;
        _bpeCache.Clear();
        _preTokenizeRegex = null;

        int idx = 0;

        // ── Шаг 1: Специальные токены ПЕРВЫМИ (строго по порядку) ──
        foreach (var st in SpecialTokensOrder)
        {
            if (!Vocab.ContainsKey(st))
            {
                Vocab[st] = idx++;
                if (verbose)
                    Console.WriteLine($"  [special] '{st}' => {idx - 1}");
            }
        }

        // ── Шаг 2: 256 байт-токенов в GPT-2 порядке ──
        // Порядок: сначала printable (33-126, 161-172, 174-255), затем остальные
        var byteOrder = new List<int>();
        for (int i = 33; i <= 126; i++) byteOrder.Add(i);
        for (int i = 161; i <= 172; i++) byteOrder.Add(i);
        for (int i = 174; i <= 255; i++) byteOrder.Add(i);
        for (int b = 0; b < 256; b++)
            if (!byteOrder.Contains(b)) byteOrder.Add(b);

        foreach (int b in byteOrder)
        {
            string tok = ByteToUnicode[(byte)b].ToString();
            if (!Vocab.ContainsKey(tok))
                Vocab[tok] = idx++;
        }

        if (verbose)
            Console.WriteLine($"Базовый vocab (specials + bytes): {idx}");

        // ── Шаг 3: Частоты слов через GPT-2 pre-tokenization ──
        var re = GetPreTokenizeRegex();

        var wordFreq = new Dictionary<string, int>();

        foreach (var line in corpus)
        {
            if (string.IsNullOrEmpty(line)) continue;
            foreach (Match m in re.Matches(line))
            {
                string piece = m.Value;

                string trimmed = piece.TrimStart();  // ← добавить
                // Спецтокены — считаем но НЕ byte-encode
                if (SpecialTokens.Contains(trimmed))  // ← trimmed вместо piece
                {
                    continue;
                }

                // Byte-encode через GPT-2 mapping
                string encoded = EncodeWordToBytes(piece);
                if (string.IsNullOrEmpty(encoded)) continue;

                wordFreq.TryGetValue(encoded, out int c);
                wordFreq[encoded] = c + 1;
            }
        }

        if (verbose)
            Console.WriteLine($"Уникальных слов: {wordFreq.Count}");

        // ── Шаг 4: Инкрементальный BPE ──
        var splits = wordFreq.Keys.ToDictionary(
            w => w,
            w => w.Select(c => c.ToString()).ToList());

        var pairFreq = new Dictionary<(string, string), int>(new PairEq());
        var pairToWords = new Dictionary<(string, string), HashSet<string>>(new PairEq());
        BuildPairIndex(splits, wordFreq, pairFreq, pairToWords);

        int maxMerges = vocabSize - Vocab.Count;
        int step = 0;

        while (step < maxMerges)
        {
            var best = FindBest(pairFreq);
            if (best == default || pairFreq.GetValueOrDefault(best, 0) < minFrequency)
                break;

            string merged = best.A + best.B;
            Merges.Add(best);
            if (!Vocab.ContainsKey(merged))
                Vocab[merged] = idx++;

            if (verbose && step % 50 == 0)
                Console.WriteLine(
                    $"step={step,4} freq={pairFreq[best],5}" +
                    $" '{Gpt2Decode(best.A)}'+'{Gpt2Decode(best.B)}'" +
                    $"=>'{Gpt2Decode(merged)}'");

            UpdateIndex(best, merged, pairToWords[best],
                        splits, wordFreq, pairFreq, pairToWords);
            step++;
        }

        sw.Stop();
        if (verbose)
            Console.WriteLine(
                $"Готово: {step} слияний, vocab={Vocab.Count} [{sw.ElapsedMilliseconds}ms]");

        // ── Шаг 5: Верификация спецтокенов ──
        VerifySpecialTokens(verbose);
    }

    /// <summary>
    /// Проверяет корректность специальных токенов в vocab.
    /// </summary>
    private void VerifySpecialTokens(bool verbose)
    {
        Console.WriteLine("\n[ВЕРИФИКАЦИЯ СПЕЦТОКЕНОВ]");
        bool ok = true;
        foreach (var st in SpecialTokensOrder)
        {
            if (Vocab.TryGetValue(st, out int id))
            {
                if (verbose)
                    Console.WriteLine($"  ✅ '{st}' => id={id}");
            }
            else
            {
                Console.WriteLine($"  ❌ '{st}' НЕ НАЙДЕН в vocab!");
                ok = false;
            }
        }

        // Проверяем порядок: <pad>=0, <unk>=1, <s>=2, </s>=3
        int[] expectedIds = { 0, 1, 2, 3, 4, 5, 6 };
        for (int i = 0; i < SpecialTokensOrder.Length; i++)
        {
            if (Vocab.TryGetValue(SpecialTokensOrder[i], out int actualId))
            {
                if (actualId != expectedIds[i])
                {
                    Console.WriteLine(
                        $"  ⚠️ '{SpecialTokensOrder[i]}' имеет id={actualId}," +
                        $" ожидалось {expectedIds[i]}");
                }
            }
        }

        if (ok) Console.WriteLine("  ✅ Все спецтокены корректны");
    }

    // ═══════════════════════════════════════════════════════
    // ENCODE
    // ═══════════════════════════════════════════════════════

    public List<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        if (_ranks == null)
        {
            _ranks = new Dictionary<(string, string), int>(new PairEq());
            for (int i = 0; i < Merges.Count; i++)
                _ranks[Merges[i]] = i;
        }

        var result = new List<int>();

        if (addBos && Vocab.TryGetValue(BosToken, out int bId))
            result.Add(bId);

        var re = GetPreTokenizeRegex();

        foreach (Match m in re.Matches(text))
        {
            string piece = m.Value;

            string trimmed = piece.TrimStart();  // ← убираем ведущий пробел
            // Спецтокены — прямой lookup по trimmed
            if (SpecialTokens.Contains(trimmed))
            {
                if (Vocab.TryGetValue(trimmed, out int sid))
                    result.Add(sid);
                else if (Vocab.TryGetValue(UnkToken, out int uk))
                    result.Add(uk);
                continue;
            }

            // Спецтокены — прямой lookup
            if (SpecialTokens.Contains(piece))
            {
                if (Vocab.TryGetValue(piece, out int sid))
                    result.Add(sid);
                else if (Vocab.TryGetValue(UnkToken, out int uk))
                    result.Add(uk);
                continue;
            }

            // Byte-encode + BPE
            string encoded = EncodeWordToBytes(piece);
            if (string.IsNullOrEmpty(encoded)) continue;

            var toks = ApplyMergesCached(encoded, _ranks);
            foreach (var t in toks)
            {
                if (Vocab.TryGetValue(t, out int tid))
                    result.Add(tid);
                else if (Vocab.TryGetValue(UnkToken, out int uk))
                    result.Add(uk);
                else
                    result.Add(UnkId);
            }
        }

        if (addEos && Vocab.TryGetValue(EosToken, out int eId))
            result.Add(eId);

        return result;
    }

    // ═══════════════════════════════════════════════════════
    // DECODE
    // ═══════════════════════════════════════════════════════

    public string Decode(List<int> ids)
    {
        var rev = Vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        var byteList = new List<byte>();

        foreach (int id in ids)
        {
            if (!rev.TryGetValue(id, out var tok)) continue;

            if (SpecialTokens.Contains(tok))
            {
                // Спецтокены декодируем как UTF-8 текст
                foreach (byte b in Encoding.UTF8.GetBytes(tok))
                    byteList.Add(b);
                continue;
            }

            // GPT-2 unicode → bytes
            foreach (char ch in tok)
                if (UnicodeToByte.TryGetValue(ch, out byte b))
                    byteList.Add(b);
        }

        return Encoding.UTF8.GetString(byteList.ToArray());
    }

    string ITokenizer.Decode(List<int> ids) => Decode(ids);

    // ═══════════════════════════════════════════════════════
    // FORMAT DIALOGUE — строго совпадает с chat_template в GGUF
    // ═══════════════════════════════════════════════════════

    /// <summary>
    /// Форматирует текст для pretraining (без спецтокенов диалогов).
    /// </summary>
    public string FormatPretraining(string text) => text.Trim();

    /// <summary>
    /// Форматирует диалог. ДОЛЖЕН совпадать с chat_template в GGUF:
    public string FormatDialogue(string user, string? assistant = null)
    {
        if (string.IsNullOrEmpty(assistant))
            return $"{UserToken} {user}{AssistantToken}";  // без пробела перед <assistant>

        return $"{UserToken} {user}{AssistantToken} {assistant}{EosToken}";
    }

    // ═══════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════

    public static string EncodeWordToBytes(string word)
    {
        var bytes = Encoding.UTF8.GetBytes(word);
        return new string(bytes.Select(b => ByteToUnicode[b]).ToArray());
    }

    public static string Gpt2Decode(string gpt2str)
    {
        var bytes = new List<byte>();
        foreach (var ch in gpt2str)
            if (UnicodeToByte.TryGetValue(ch, out byte b)) bytes.Add(b);
        try { return Encoding.UTF8.GetString(bytes.ToArray()); }
        catch { return gpt2str; }
    }

    private List<string> ApplyMergesCached(
        string encoded,
        Dictionary<(string, string), int> ranks)
    {
        if (_bpeCache.TryGetValue(encoded, out var cached))
            return cached.ToList();

        var toks = encoded.Select(c => c.ToString()).ToList();
        while (toks.Count > 1)
        {
            int best = int.MaxValue, bi = -1;
            for (int i = 0; i < toks.Count - 1; i++)
                if (ranks.TryGetValue((toks[i], toks[i + 1]), out int r) && r < best)
                { best = r; bi = i; }

            if (bi < 0) break;
            toks[bi] = toks[bi] + toks[bi + 1];
            toks.RemoveAt(bi + 1);
        }

        _bpeCache[encoded] = toks.ToArray();
        return toks;
    }

    // ═══════════════════════════════════════════════════════
    // BPE internals
    // ═══════════════════════════════════════════════════════

    private static void BuildPairIndex(
        Dictionary<string, List<string>> splits,
        Dictionary<string, int> wf,
        Dictionary<(string, string), int> pf,
        Dictionary<(string, string), HashSet<string>> pw)
    {
        foreach (var (w, freq) in wf)
        {
            var p = splits[w];
            for (int i = 0; i < p.Count - 1; i++)
            {
                var pair = (p[i], p[i + 1]);
                pf.TryGetValue(pair, out int c);
                pf[pair] = c + freq;
                if (!pw.TryGetValue(pair, out var ws))
                    pw[pair] = ws = new HashSet<string>();
                ws.Add(w);
            }
        }
    }

    private static (string A, string B) FindBest(
        Dictionary<(string, string), int> pf)
    {
        (string A, string B) best = default;
        int bestFreq = -1;
        foreach (var kv in pf)
            if (kv.Value > bestFreq ||
                kv.Value == bestFreq &&
                string.Compare(kv.Key.Item1 + kv.Key.Item2,
                               best.A + best.B,
                               StringComparison.Ordinal) < 0)
            { best = (kv.Key.Item1, kv.Key.Item2); bestFreq = kv.Value; }
        return best;
    }

    private static void UpdateIndex(
        (string A, string B) mp,
        string merged,
        HashSet<string> affected,
        Dictionary<string, List<string>> splits,
        Dictionary<string, int> wf,
        Dictionary<(string, string), int> pf,
        Dictionary<(string, string), HashSet<string>> pw)
    {
        pf.Remove(mp);
        pw.Remove(mp);

        foreach (var word in affected.ToList())
        {
            int freq = wf[word];
            var old = splits[word];

            // Убираем старые пары
            for (int i = 0; i < old.Count - 1; i++)
            {
                var pair = (old[i], old[i + 1]);
                if (pair == mp) continue;
                if (pf.ContainsKey(pair))
                {
                    pf[pair] -= freq;
                    if (pf[pair] <= 0) { pf.Remove(pair); pw.Remove(pair); }
                    else pw[pair]?.Remove(word);
                }
            }

            // Применяем слияние
            var newPt = ApplyOnce(old, mp.A, mp.B, merged);
            splits[word] = newPt;

            // Добавляем новые пары
            for (int i = 0; i < newPt.Count - 1; i++)
            {
                var pair = (newPt[i], newPt[i + 1]);
                pf.TryGetValue(pair, out int c);
                pf[pair] = c + freq;
                if (!pw.TryGetValue(pair, out var ws))
                    pw[pair] = ws = new HashSet<string>();
                ws.Add(word);
            }
        }
    }

    private static List<string> ApplyOnce(
        List<string> p, string a, string b, string m)
    {
        var r = new List<string>(p.Count);
        int i = 0;
        while (i < p.Count)
        {
            if (i < p.Count - 1 && p[i] == a && p[i + 1] == b)
            { r.Add(m); i += 2; }
            else
                r.Add(p[i++]);
        }
        return r;
    }

    private class PairEq : IEqualityComparer<(string, string)>
    {
        public bool Equals((string, string) x, (string, string) y)
            => x.Item1 == y.Item1 && x.Item2 == y.Item2;
        public int GetHashCode((string, string) o)
            => HashCode.Combine(o.Item1, o.Item2);
    }

    // ═══════════════════════════════════════════════════════
    // СОХРАНЕНИЕ / ЗАГРУЗКА
    // ═══════════════════════════════════════════════════════

    public Vocab ToVocab()
    {
        var vocabList = Vocab.OrderBy(kv => kv.Value).ToList();
        var words = new List<string>(new string[vocabList.Count]);
        foreach (var (token, id) in vocabList)
            words[id] = token;
        return new Vocab(words);
    }

    void ITokenizer.SaveToStream(BinaryWriter writer)
    {
        writer.Write("BPE2FIXED_V1");
        writer.Write(SpecialTokens.Count);
        foreach (var s in SpecialTokens) writer.Write(s);

        var vocabList = Vocab.OrderBy(kv => kv.Value).ToList();
        writer.Write(vocabList.Count);
        foreach (var (token, id) in vocabList)
        {
            writer.Write(token);
            writer.Write(id);
        }

        writer.Write(Merges.Count);
        foreach (var (a, b) in Merges)
        {
            writer.Write(a);
            writer.Write(b);
        }
    }

    /// <summary>
    /// Загружает токенизатор из BinaryReader (изнутри модели).
    /// </summary>
    public static BPETokenizer LoadFromStream(BinaryReader reader)
    {
        var tokenizer = new BPETokenizer();

        string marker = reader.ReadString();
        if (marker != "BPE2FIXED_V1")
            throw new InvalidDataException(
                $"Неверный формат токенизатора: {marker}");

        // Special tokens
        int specialCount = reader.ReadInt32();
        tokenizer.SpecialTokens = new List<string>(specialCount);
        for (int i = 0; i < specialCount; i++)
            tokenizer.SpecialTokens.Add(reader.ReadString());

        // Vocab
        int vocabSize = reader.ReadInt32();
        tokenizer.Vocab = new Dictionary<string, int>(vocabSize);
        for (int i = 0; i < vocabSize; i++)
        {
            string token = reader.ReadString();
            int id = reader.ReadInt32();
            tokenizer.Vocab[token] = id;
        }

        // Merges
        int mergeCount = reader.ReadInt32();
        tokenizer.Merges = new List<(string, string)>(mergeCount);
        for (int i = 0; i < mergeCount; i++)
        {
            string a = reader.ReadString();
            string b = reader.ReadString();
            tokenizer.Merges.Add((a, b));
        }

        return tokenizer;
    }

    public class TokenizerJson
    {
        public Dictionary<string, int> Vocab { get; set; } = new();
        public List<string> Merges { get; set; } = new();
        public List<string> SpecialTokens { get; set; } = new();
    }
}