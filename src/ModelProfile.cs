//https://github.com/virex-84

using System.Text;

namespace LLM.ILGPU;

/// <summary>
/// Тип модели — определяет базовые пропорции архитектуры.
/// </summary>
public enum ModelProfile
{
    /// <summary>
    /// Быстрое обучение, малый корпус (50-200 примеров).
    /// Подходит для прототипирования и тестов.
    /// </summary>
    Small,

    /// <summary>
    /// Баланс качества и скорости (200-1000 примеров).
    /// Подходит для небольших доменных задач.
    /// </summary>
    Medium,

    /// <summary>
    /// Максимальное запоминание фактов (500-5000+ примеров).
    /// Для задач, требующих точного воспроизведения знаний.
    /// </summary>
    Large,

    /// <summary>
    /// Автоматический выбор на основе анализа корпуса.
    /// </summary>
    Auto
}

/// <summary>
/// Конфигурация модели — все гиперпараметры в одном месте.
/// </summary>
public class ModelConfig
{
    // ─── Архитектура ───
    public int EmbeddingDim { get; set; }
    public int HiddenDim { get; set; }
    public int NumHeads { get; set; }
    public int NumLayers { get; set; }
    public int MaxSeqLen { get; set; }
    public int BpeVocabSize { get; set; }

    // ─── Обучение: Pretrain ───
    public int PretrainEpochs { get; set; }
    public float PretrainLr { get; set; }
    public int PretrainWarmupSteps { get; set; }

    // ─── Обучение: Finetune ───
    public int FinetuneEpochs { get; set; }
    public float FinetuneLr { get; set; }
    public int FinetuneWarmupSteps { get; set; }

    // ─── Общие ───
    public float GradientClipNorm { get; set; }
    public ModelProfile Profile { get; set; }

    // ─── Метаданные (заполняются анализатором) ───
    public int EstimatedParameters { get; set; }
    public float EstimatedMemoryMB { get; set; }
    public int RecommendedMinPretrainSamples { get; set; }
    public int RecommendedMinFinetuneSamples { get; set; }

    /// <summary>
    /// Оценка количества параметров (приблизительная).
    /// </summary>
    public int ComputeEstimatedParameters()
    {
        int embParams = BpeVocabSize * EmbeddingDim;
        int attnParams = 4 * EmbeddingDim * EmbeddingDim; // Wq,Wk,Wv,Wo
        int ffnParams = 2 * EmbeddingDim * HiddenDim + HiddenDim + EmbeddingDim;
        int lnParams = 2 * EmbeddingDim; // gamma + beta
        int blockParams = attnParams + ffnParams + 2 * lnParams;
        int outputParams = EmbeddingDim * BpeVocabSize + BpeVocabSize;

        EstimatedParameters = embParams + NumLayers * blockParams + outputParams;
        EstimatedMemoryMB = EstimatedParameters * 4f / (1024f * 1024f) * 3f;
        // ×3: параметры + градиенты + оптимизатор (m,v)

        return EstimatedParameters;
    }
}

/// <summary>
/// Статистика корпуса — результат анализа данных.
/// </summary>
public class CorpusStats
{
    // ─── Размеры ───
    public int PretrainSampleCount { get; set; }
    public int FinetuneSampleCount { get; set; }
    public int TotalSampleCount => PretrainSampleCount + FinetuneSampleCount;

    // ─── Длины ───
    public float AvgTokensPerSample { get; set; }
    public int MaxTokensInSample { get; set; }
    public int MinTokensInSample { get; set; }
    public int MedianTokensPerSample { get; set; }
    public int Percentile95Tokens { get; set; }

    // ─── Словарь ───
    public int UniqueWords { get; set; }
    public int UniqueChars { get; set; }
    public int TotalWords { get; set; }
    public float AvgWordLength { get; set; }
    public float TypeTokenRatio { get; set; } // unique/total

    // ─── Структура ───
    public int SamplesWithUserAssistant { get; set; }
    public int SamplesWithEos { get; set; }
    public float AvgWordsPerSample { get; set; }
    public int LongestWordLength { get; set; }

    // ─── Языковые характеристики ───
    public bool HasCyrillic { get; set; }
    public bool HasLatin { get; set; }
    public float CyrillicRatio { get; set; }
    public int PunctuationTypes { get; set; }
}

/// <summary>
/// Анализатор корпуса и генератор конфигурации.
/// Анализирует данные, выбирает оптимальные гиперпараметры,
/// выдаёт рекомендации по улучшению.
/// </summary>
public class ModelConfigBuilder
{
    // ═══════════════════════════════════════════════════════════
    // Базовые профили (hard-coded defaults)
    // ═══════════════════════════════════════════════════════════

    private static readonly Dictionary<ModelProfile, ModelConfig> BaseProfiles = new()
    {
        [ModelProfile.Small] = new ModelConfig
        {
            EmbeddingDim = 64,
            HiddenDim = 128,
            NumHeads = 2,
            NumLayers = 2,
            MaxSeqLen = 64,
            BpeVocabSize = 500,
            PretrainEpochs = 30,
            PretrainLr = 2e-3f,
            PretrainWarmupSteps = 50,
            FinetuneEpochs = 20,
            FinetuneLr = 5e-4f,
            FinetuneWarmupSteps = 30,
            GradientClipNorm = 5.0f,
            Profile = ModelProfile.Small,
            RecommendedMinPretrainSamples = 50,
            RecommendedMinFinetuneSamples = 20,
        },
        [ModelProfile.Medium] = new ModelConfig
        {
            EmbeddingDim = 128,
            HiddenDim = 256,
            NumHeads = 4,
            NumLayers = 3,
            MaxSeqLen = 80,
            BpeVocabSize = 1000,
            PretrainEpochs = 80,
            PretrainLr = 1e-3f,
            PretrainWarmupSteps = 100,
            FinetuneEpochs = 40,
            FinetuneLr = 3e-4f,
            FinetuneWarmupSteps = 50,
            GradientClipNorm = 5.0f,
            Profile = ModelProfile.Medium,
            RecommendedMinPretrainSamples = 200,
            RecommendedMinFinetuneSamples = 50,
        },
        [ModelProfile.Large] = new ModelConfig
        {
            EmbeddingDim = 256,
            HiddenDim = 512,
            NumHeads = 8,
            NumLayers = 5,
            MaxSeqLen = 128,
            BpeVocabSize = 3000,
            PretrainEpochs = 200,
            PretrainLr = 1e-3f,
            PretrainWarmupSteps = 200,
            FinetuneEpochs = 80,
            FinetuneLr = 1e-4f,
            FinetuneWarmupSteps = 100,
            GradientClipNorm = 3.0f,
            Profile = ModelProfile.Large,
            RecommendedMinPretrainSamples = 500,
            RecommendedMinFinetuneSamples = 100,
        }
    };

    // ═══════════════════════════════════════════════════════════
    // АНАЛИЗ КОРПУСА
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Анализирует корпус и возвращает статистику.
    /// </summary>
    public static CorpusStats AnalyzeCorpus(
        List<string> pretrainData, List<string> finetuneData)
    {
        var stats = new CorpusStats
        {
            PretrainSampleCount = pretrainData.Count,
            FinetuneSampleCount = finetuneData.Count,
        };

        var allTexts = pretrainData.Concat(finetuneData).ToList();
        if (allTexts.Count == 0) return stats;

        // ─── Разбиваем на слова ───
        var allWords = new List<string>();
        var uniqueWords = new HashSet<string>();
        var uniqueChars = new HashSet<char>();
        var punctuationChars = new HashSet<char>();
        var wordLengths = new List<int>();
        var sampleWordCounts = new List<int>();
        int cyrillicChars = 0;
        int latinChars = 0;
        int totalChars = 0;

        foreach (var text in allTexts)
        {
            string clean = text.Replace("</s>", "")
                .Replace("<bos>", "").Replace("<user>", "")
                .Replace("<assistant>", "").Replace("<sep>", "").Trim();

            var words = clean.Split((char[])null!,
                StringSplitOptions.RemoveEmptyEntries);

            sampleWordCounts.Add(words.Length);

            foreach (var word in words)
            {
                allWords.Add(word);
                uniqueWords.Add(word.ToLowerInvariant());
                wordLengths.Add(word.Length);

                foreach (char c in word)
                {
                    uniqueChars.Add(c);
                    totalChars++;

                    if (c >= 'а' && c <= 'я' || c >= 'А' && c <= 'Я' || c == 'ё' || c == 'Ё')
                        cyrillicChars++;
                    else if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z')
                        latinChars++;

                    if (char.IsPunctuation(c))
                        punctuationChars.Add(c);
                }
            }

            // Структура
            if (text.Contains("User:", StringComparison.OrdinalIgnoreCase) &&
                text.Contains("Assistant:", StringComparison.OrdinalIgnoreCase))
                stats.SamplesWithUserAssistant++;

            if (text.Contains("</s>"))
                stats.SamplesWithEos++;
        }

        // ─── Заполняем статистику ───
        stats.UniqueWords = uniqueWords.Count;
        stats.UniqueChars = uniqueChars.Count;
        stats.TotalWords = allWords.Count;
        stats.AvgWordLength = wordLengths.Count > 0
            ? (float)wordLengths.Average() : 0;
        stats.LongestWordLength = wordLengths.Count > 0
            ? wordLengths.Max() : 0;
        stats.TypeTokenRatio = allWords.Count > 0
            ? (float)uniqueWords.Count / allWords.Count : 0;

        stats.AvgWordsPerSample = sampleWordCounts.Count > 0
            ? (float)sampleWordCounts.Average() : 0;

        // Оценка длин в токенах (≈ 1.3× от слов для BPE русского)
        float bpeMultiplier = (cyrillicChars > latinChars) ? 1.5f : 1.2f;
        var tokenEstimates = sampleWordCounts
            .Select(wc => (int)(wc * bpeMultiplier))
            .OrderBy(x => x).ToList();

        if (tokenEstimates.Count > 0)
        {
            stats.AvgTokensPerSample = (float)tokenEstimates.Average();
            stats.MinTokensInSample = tokenEstimates.First();
            stats.MaxTokensInSample = tokenEstimates.Last();
            stats.MedianTokensPerSample =
                tokenEstimates[tokenEstimates.Count / 2];
            stats.Percentile95Tokens =
                tokenEstimates[(int)(tokenEstimates.Count * 0.95)];
        }

        stats.HasCyrillic = cyrillicChars > 0;
        stats.HasLatin = latinChars > 0;
        stats.CyrillicRatio = totalChars > 0
            ? (float)cyrillicChars / totalChars : 0;
        stats.PunctuationTypes = punctuationChars.Count;

        return stats;
    }

    // ═══════════════════════════════════════════════════════════
    // ГЕНЕРАЦИЯ КОНФИГУРАЦИИ
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Создаёт конфигурацию на основе профиля + адаптирует под корпус.
    /// </summary>
    public static ModelConfig CreateConfig(
        ModelProfile profile, CorpusStats stats)
    {
        // Если Auto — определяем профиль автоматически
        if (profile == ModelProfile.Auto)
            profile = DetermineAutoProfile(stats);

        var config = CloneConfig(BaseProfiles[profile]);

        // ─── Адаптация под данные ───
        AdaptToCorpus(config, stats);

        config.ComputeEstimatedParameters();
        return config;
    }

    // ═══════════════════════════════════════════════════════════
    // АВТООПРЕДЕЛЕНИЕ ПРОФИЛЯ
    // ═══════════════════════════════════════════════════════════

    private static ModelProfile DetermineAutoProfile(CorpusStats stats)
    {
        int totalSamples = stats.TotalSampleCount;
        int uniqueWords = stats.UniqueWords;

        // Критерии выбора:
        // 1. Количество данных
        // 2. Сложность словаря
        // 3. Длина последовательностей

        // Мало данных или маленький словарь → Small
        if (totalSamples < 100 || uniqueWords < 200)
            return ModelProfile.Small;

        // Много данных и большой словарь → Large
        if (totalSamples >= 500 && uniqueWords >= 1000)
            return ModelProfile.Large;

        // Средние длины > 50 токенов + достаточно данных → Large
        if (stats.AvgTokensPerSample > 50 && totalSamples >= 300)
            return ModelProfile.Large;

        return ModelProfile.Medium;
    }

    // ═══════════════════════════════════════════════════════════
    // АДАПТАЦИЯ ПОД КОРПУС
    // ═══════════════════════════════════════════════════════════

    private static void AdaptToCorpus(ModelConfig config, CorpusStats stats)
    {
        // ─── MaxSeqLen: по 95-му перцентилю + запас ───
        if (stats.Percentile95Tokens > 0)
        {
            int adaptedMaxSeq = Math.Min(
                (int)(stats.Percentile95Tokens * 1.3f + 10),
                256); // Верхний предел
            // Округляем до кратного 8
            adaptedMaxSeq = ((adaptedMaxSeq + 7) / 8) * 8;
            config.MaxSeqLen = Math.Max(config.MaxSeqLen, adaptedMaxSeq);
        }

        // ─── BPE vocab size: по количеству уникальных слов ───
        if (stats.UniqueWords > 0)
        {
            // BPE обычно нужно 1.5-3× от уникальных слов
            // (подслова дробят, но также добавляются объединения)
            int adaptedVocab = stats.UniqueWords;

            // Для кириллицы нужен больший словарь
            // (больше морфологических форм)
            if (stats.CyrillicRatio > 0.5f)
                adaptedVocab = (int)(adaptedVocab * 1.5f);

            // Ограничиваем разумным диапазоном
            adaptedVocab = Math.Clamp(adaptedVocab,
                config.BpeVocabSize / 2,   // не меньше половины базового
                config.BpeVocabSize * 3);  // не больше тройного базового

            // Округляем до кратного 50
            config.BpeVocabSize = ((adaptedVocab + 49) / 50) * 50;
        }

        // ─── Эпохи: обратно пропорциональны количеству данных ───
        if (stats.TotalSampleCount > 0)
        {
            // Формула: больше данных → меньше эпох (но не меньше 10)
            float dataFactor = 500f / Math.Max(stats.TotalSampleCount, 1);
            dataFactor = Math.Clamp(dataFactor, 0.3f, 3.0f);

            config.PretrainEpochs = Math.Max(10,
                (int)(config.PretrainEpochs * dataFactor));
            config.FinetuneEpochs = Math.Max(5,
                (int)(config.FinetuneEpochs * dataFactor));
        }

        // ─── Warmup: 5-10% от общего количества шагов ───
        int pretrainSteps = config.PretrainEpochs *
            Math.Max(stats.PretrainSampleCount, 1);
        config.PretrainWarmupSteps = Math.Clamp(
            pretrainSteps / 15, 10, 500);

        int finetuneSteps = config.FinetuneEpochs *
            Math.Max(stats.FinetuneSampleCount, 1);
        config.FinetuneWarmupSteps = Math.Clamp(
            finetuneSteps / 15, 5, 200);

        // ─── LR: для больших датасетов можно выше ───
        if (stats.TotalSampleCount > 1000)
        {
            config.PretrainLr *= 0.7f; // Чуть ниже для стабильности
        }

        // ─── Gradient clip: жёстче для малых датасетов ───
        if (stats.TotalSampleCount < 100)
        {
            config.GradientClipNorm = 2.0f;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // УТИЛИТЫ
    // ═══════════════════════════════════════════════════════════

    private static ModelConfig CloneConfig(ModelConfig source)
    {
        return new ModelConfig
        {
            EmbeddingDim = source.EmbeddingDim,
            HiddenDim = source.HiddenDim,
            NumHeads = source.NumHeads,
            NumLayers = source.NumLayers,
            MaxSeqLen = source.MaxSeqLen,
            BpeVocabSize = source.BpeVocabSize,
            PretrainEpochs = source.PretrainEpochs,
            PretrainLr = source.PretrainLr,
            PretrainWarmupSteps = source.PretrainWarmupSteps,
            FinetuneEpochs = source.FinetuneEpochs,
            FinetuneLr = source.FinetuneLr,
            FinetuneWarmupSteps = source.FinetuneWarmupSteps,
            GradientClipNorm = source.GradientClipNorm,
            Profile = source.Profile,
            RecommendedMinPretrainSamples = source.RecommendedMinPretrainSamples,
            RecommendedMinFinetuneSamples = source.RecommendedMinFinetuneSamples,
        };
    }
}