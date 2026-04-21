//https://github.com/virex-84

namespace LLM.ILGPU;

public class ModelConfig
{
    // ── Архитектура ──────────────────────────────────────────
    public int EmbeddingDim { get; set; }
    public int HiddenDim { get; set; }
    public int NumHeads { get; set; }
    public int NumLayers { get; set; }
    public int MaxSeqLen { get; set; }
    public int BpeVocabSize { get; set; }

    // ── Pretrain ─────────────────────────────────────────────
    public int PretrainEpochs { get; set; }
    public float PretrainLr { get; set; }
    public int PretrainWarmupSteps { get; set; }

    // ── Finetune ─────────────────────────────────────────────
    public int FinetuneEpochs { get; set; }
    public float FinetuneLr { get; set; }
    public int FinetuneWarmupSteps { get; set; }

    // ── Общие ────────────────────────────────────────────────
    public float GradientClipNorm { get; set; }

    // ── Метаданные ───────────────────────────────────────────
    public int EstimatedParameters { get; set; }
    public float EstimatedMemoryMB { get; set; }

    public int ComputeEstimatedParameters()
    {
        int embParams = BpeVocabSize * EmbeddingDim;
        int attnParams = 4 * EmbeddingDim * EmbeddingDim;
        int ffnParams = 2 * EmbeddingDim * HiddenDim + HiddenDim + EmbeddingDim;
        int lnParams = 2 * EmbeddingDim;
        int blockParams = attnParams + ffnParams + 2 * lnParams;
        int outputParams = EmbeddingDim * BpeVocabSize + BpeVocabSize;

        EstimatedParameters = embParams + NumLayers * blockParams + outputParams;
        EstimatedMemoryMB = EstimatedParameters * 4f / (1024f * 1024f) * 3f;
        return EstimatedParameters;
    }

    // ═══════════════════════════════════════════════════════════
    // РЕКОМЕНДАЦИИ
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Выводит в консоль подробные рекомендации по конфигурации модели:
    /// — предупреждения о несоответствиях корпуса и архитектуры
    /// — описание реальных возможностей модели с текущими параметрами
    /// — советы по улучшению
    /// Вызывается после BuildOptimalConfig при создании новой модели.
    /// </summary>
    public void PrintRecommendations(CorpusStats stats)
    {
        var w = Console.Out;
        var cc = Console.ForegroundColor;

        void Header(string text)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            w.WriteLine($"\n╔══════════════════════════════════════════════════╗");
            w.WriteLine($"║  {text,-48}║");
            w.WriteLine($"╚══════════════════════════════════════════════════╝");
            Console.ForegroundColor = cc;
        }

        void Warn(string text)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            w.WriteLine($"  ⚠  {text}");
            Console.ForegroundColor = cc;
        }

        void Ok(string text)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            w.WriteLine($"  ✓  {text}");
            Console.ForegroundColor = cc;
        }

        void Info(string text)
        {
            Console.ForegroundColor = ConsoleColor.Gray;
            w.WriteLine($"  •  {text}");
            Console.ForegroundColor = cc;
        }

        void Tip(string text)
        {
            Console.ForegroundColor = ConsoleColor.DarkCyan;
            w.WriteLine($"  💡 {text}");
            Console.ForegroundColor = cc;
        }

        void Section(string text)
        {
            Console.ForegroundColor = ConsoleColor.White;
            w.WriteLine($"\n  ── {text} ──");
            Console.ForegroundColor = cc;
        }

        // ═══════════════════════════════════════════════════════
        // 1. ПРЕДУПРЕЖДЕНИЯ О НЕСООТВЕТСТВИЯХ
        // ═══════════════════════════════════════════════════════
        Header("АНАЛИЗ КОРПУСА И КОНФИГУРАЦИИ");

        bool hasWarnings = false;

        // --- Дисбаланс pretrain / finetune ---
        if (stats.PretrainSampleCount > 0 && stats.FinetuneSampleCount > 0)
        {
            float ratio = (float)stats.FinetuneSampleCount / stats.PretrainSampleCount;
            if (ratio > 20f)
            {
                Warn($"Критический дисбаланс: finetune в {ratio:F0}× больше pretrain " +
                     $"({stats.PretrainSampleCount} vs {stats.FinetuneSampleCount}).");
                Warn("Модель не успеет выучить базовый язык на pretrain " +
                     "и будет «ломаться» на finetune.");
                Tip("Добавьте хотя бы 200-500 pretrain примеров, " +
                    "или увеличьте PretrainEpochs вручную до 300+.");
                hasWarnings = true;
            }
            else if (ratio > 5f)
            {
                Warn($"Заметный дисбаланс: finetune в {ratio:F0}× больше pretrain. " +
                     "Базовые языковые паттерны могут быть слабыми.");
                Tip("Рекомендуется pretrain / finetune ≈ 1:3 или лучше.");
                hasWarnings = true;
            }
            else
            {
                Ok($"Баланс корпуса приемлем: pretrain={stats.PretrainSampleCount}, " +
                   $"finetune={stats.FinetuneSampleCount}.");
            }
        }

        // --- Очень мало pretrain ---
        if (stats.PretrainSampleCount < 50)
        {
            Warn($"Критически мало pretrain примеров: {stats.PretrainSampleCount}. " +
                 "Модель не сможет усвоить базовую структуру языка.");
            Tip("Минимум для осмысленного pretrain: 200+ примеров.");
            hasWarnings = true;
        }

        // --- VocabSize vs корпус ---
        if (BpeVocabSize > 2048 && stats.TotalSampleCount < 500)
        {
            Warn($"Словарь BPE ({BpeVocabSize} токенов) слишком велик " +
                 $"для корпуса из {stats.TotalSampleCount} примеров.");
            Warn("Большинство токенов не встретятся достаточно часто — " +
                 "эмбеддинги не обучатся.");
            Tip($"Оптимальный vocab для {stats.TotalSampleCount} примеров: " +
                $"256–1024 токенов.");
            hasWarnings = true;
        }
        else if (BpeVocabSize < 512 && stats.UniqueWords > 1000)
        {
            Warn($"Словарь BPE ({BpeVocabSize}) мал для {stats.UniqueWords} " +
                 "уникальных слов. Слова будут дроблены на слишком мелкие части.");
            hasWarnings = true;
        }
        else
        {
            Ok($"Размер словаря {BpeVocabSize} соответствует объёму корпуса.");
        }

        // --- MaxSeqLen ---
        if (MaxSeqLen < 64)
        {
            Warn($"MaxSeqLen={MaxSeqLen} очень мало. " +
                 "Большинство примеров будут обрезаны.");
            hasWarnings = true;
        }
        else
        {
            Ok($"MaxSeqLen={MaxSeqLen} покрывает реальные длины корпуса.");
        }

        // --- NumLayers ---
        if (NumLayers == 1 && stats.FinetuneSampleCount > 500)
        {
            Warn("1 трансформерный слой не способен улавливать " +
                 "многоуровневые паттерны диалога.");
            Tip("Для chat-данных рекомендуется NumLayers ≥ 2.");
            hasWarnings = true;
        }

        // --- EmbeddingDim ---
        if (EmbeddingDim < 128 && stats.UniqueWords > 500)
        {
            Warn($"EmbeddingDim={EmbeddingDim} мало для словаря " +
                 $"из {stats.UniqueWords} слов. " +
                 "Пространство представлений будет перегружено.");
            hasWarnings = true;
        }

        // --- Кириллица ---
        if (stats.CyrillicRatio > 0.5f && BpeVocabSize < 512)
        {
            Warn("Русскоязычный корпус требует увеличенный словарь BPE. " +
                 "Рекомендуется BpeVocabSize ≥ 512.");
            hasWarnings = true;
        }

        if (!hasWarnings)
            Ok("Все параметры конфигурации выглядят сбалансированно.");

        // ═══════════════════════════════════════════════════════
        // 2. ВОЗМОЖНОСТИ ТЕКУЩЕЙ МОДЕЛИ
        // ═══════════════════════════════════════════════════════
        Header($"ВОЗМОЖНОСТИ МОДЕЛИ ({EstimatedParameters:N0} параметров)");

        string capability = GetCapabilityLevel();
        string capColor = capability switch
        {
            "memorizer" => "красный  — заучивание",
            "ngram" => "жёлтый   — N-gram уровень",
            "sentence" => "синий    — уровень предложения",
            "dialog" => "зелёный  — диалог",
            _ => "серый"
        };

        Console.ForegroundColor = capability switch
        {
            "memorizer" => ConsoleColor.Red,
            "ngram" => ConsoleColor.Yellow,
            "sentence" => ConsoleColor.Cyan,
            "dialog" => ConsoleColor.Green,
            _ => ConsoleColor.Gray,
        };
        w.WriteLine($"\n  Уровень: {capColor}");
        Console.ForegroundColor = cc;

        switch (capability)
        {
            // ── ЗАУЧИВАТЕЛЬ ──────────────────────────────────
            case "memorizer":
                Section("Что умеет");
                Ok("Мгновенное обучение (секунды на CPU).");
                Ok("Воспроизводит фразы из обучающего набора почти дословно.");
                Ok("Файл модели занимает менее 1 МБ.");
                Ok("Идеальна для отладки кода и проверки пайплайна.");

                Section("Чего НЕ умеет");
                Warn("Не понимает смысл — работает как «умный Т9».");
                Warn("Не обобщает: незнакомая фраза → мусорный вывод.");
                Warn($"Память {MaxSeqLen} токенов при 1 слое " +
                     "— «забывает» начало предложения.");
                Warn($"Словарь {BpeVocabSize} токенов " +
                     "дробит слова на буквы/слоги, теряя смысл.");
                Warn("Не способна вести диалог: роли user/assistant смешиваются.");

                Section("Практическое применение");
                Info("Автодополнение коротких фраз из фиксированного словаря.");
                Info("Шаблонные ответы типа FAQ (если ответы точно есть в train).");
                Info("Тест инфраструктуры: GPU, токенизатор, формат данных.");
                break;

            // ── N-GRAM УРОВЕНЬ ───────────────────────────────
            case "ngram":
                Section("Что умеет");
                Ok("Строит статистически правдоподобные продолжения фраз.");
                Ok("Улавливает простые биграммы и триграммы.");
                Ok("Быстрое обучение (минуты).");
                Ok("Хорошо работает как «автодополнение» в узкой предметной области.");

                Section("Чего НЕ умеет");
                Warn("Не понимает грамматику — предложения могут быть бессвязными.");
                Warn("Не держит контекст длиннее 2-3 слов.");
                Warn("При выходе за пределы обучающего распределения — мусор.");
                Warn("Не способна отличить вопрос от утверждения.");

                Section("Практическое применение");
                Info("Генерация коротких описаний товаров/услуг по шаблону.");
                Info("Простое автодополнение поисковых запросов.");
                Info("Лёгкий чат-бот с жёстко ограниченным сценарием.");
                break;

            // ── УРОВЕНЬ ПРЕДЛОЖЕНИЯ ──────────────────────────
            case "sentence":
                Section("Что умеет");
                Ok("Строит грамматически связные предложения.");
                Ok("Улавливает связь «подлежащее → сказуемое → дополнение».");
                Ok("Удерживает тему внутри одного абзаца.");
                Ok("Понимает базовые инструкции в стиле «сделай X».");
                Ok("Словарь достаточен, чтобы слова не дробились на слоги.");

                Section("Чего НЕ умеет");
                Warn("Диалог из нескольких реплик — теряет нить разговора.");
                Warn("Логические цепочки из 3+ шагов.");
                Warn("Роль персонажа держит неустойчиво.");

                Section("Практическое применение");
                Info("Генерация описаний, аннотаций, коротких текстов.");
                Info("Простой Q&A-бот по документу (без глубокого рассуждения).");
                Info("Автоматизация коротких текстовых задач в пайплайне.");
                break;

            // ── ДИАЛОГ ──────────────────────────────────────
            case "dialog":
                Section("Что умеет");
                Ok("Ведёт многоходовой диалог, сохраняя контекст.");
                Ok("Следует инструкциям и удерживает роль (при instruct-тюнинге).");
                Ok("Строит логически связные ответы из нескольких предложений.");
                Ok("Понимает смысл вопроса и отвечает по существу.");
                Ok("Способна к базовому рассуждению и обобщению.");

                Section("Практическое применение");
                Info("Полноценный чат-бот для узкой предметной области.");
                Info("Ассистент с поддержкой ролей (user/system/assistant).");
                Info("Генерация структурированного контента (отчёты, письма).");
                break;
        }

        // ═══════════════════════════════════════════════════════
        // 3. СОВЕТЫ ПО УЛУЧШЕНИЮ
        // ═══════════════════════════════════════════════════════
        Header("КАК УЛУЧШИТЬ МОДЕЛЬ");

        PrintUpgradePath(stats);

        w.WriteLine();
    }

    // ───────────────────────────────────────────────────────────
    // Определяет уровень возможностей по архитектуре
    // ───────────────────────────────────────────────────────────
    private string GetCapabilityLevel()
    {
        // dialog: достаточно слоёв, embedding и vocab для диалога
        if (NumLayers >= 4 && EmbeddingDim >= 192 &&
            BpeVocabSize >= 2048 && MaxSeqLen >= 128)
            return "dialog";

        // sentence: умеренная архитектура
        if (NumLayers >= 2 && EmbeddingDim >= 128 &&
            BpeVocabSize >= 1024 && MaxSeqLen >= 96)
            return "sentence";

        // ngram: хотя бы что-то осмысленное
        if (EmbeddingDim >= 64 && BpeVocabSize >= 256 && MaxSeqLen >= 48)
            return "ngram";

        return "memorizer";
    }

    // ───────────────────────────────────────────────────────────
    // Советы по улучшению под конкретный corpus
    // ───────────────────────────────────────────────────────────
    private void PrintUpgradePath(CorpusStats stats)
    {
        void Tip(string text)
        {
            var cc = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.DarkCyan;
            Console.WriteLine($"  💡 {text}");
            Console.ForegroundColor = cc;
        }

        void Config(string label, string value)
        {
            var cc = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"     {label,-20}");
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(value);
            Console.ForegroundColor = cc;
        }

        string level = GetCapabilityLevel();

        if (level == "memorizer" || level == "ngram")
        {
            Console.WriteLine(
                "\n  Минимум для понимания предложений (~1-2 млн параметров):");
            Config("EmbeddingDim:", "128");
            Config("HiddenDim:", "512");
            Config("NumHeads:", "4");
            Config("NumLayers:", "2");
            Config("BpeVocabSize:", "1024–2048");
            Config("MaxSeqLen:", "128");
            Tip("Для этого нужно минимум 500+ примеров pretrain.");
        }

        if (level != "dialog")
        {
            Console.WriteLine(
                "\n  Минимум для полноценного чат-бота (~5-15 млн параметров):");
            Config("EmbeddingDim:", "256");
            Config("HiddenDim:", "1024");
            Config("NumHeads:", "8");
            Config("NumLayers:", "4–6");
            Config("BpeVocabSize:", "4096–8192");
            Config("MaxSeqLen:", "256–512");
            Tip("Для этого нужно минимум 2000+ pretrain и 5000+ finetune примеров.");
        }

        // Специфичные советы по корпусу
        Console.WriteLine();

        if (stats.PretrainSampleCount < 100)
            Tip($"Увеличьте pretrain корпус: сейчас {stats.PretrainSampleCount} " +
                "примеров. Добавьте общие тексты на том же языке.");

        if (stats.CyrillicRatio > 0.5f)
            Tip("Для русского языка BpeVocabSize < 1024 — " +
                "слова будут разбиты на отдельные буквы.");

        if (stats.FinetuneSampleCount > 0 && NumLayers < 2)
            Tip("Для instruct-tuning нужно минимум 2 слоя, " +
                "иначе модель не различает роли user/assistant.");

        float imbalanceRatio = stats.PretrainSampleCount > 0
            ? (float)stats.FinetuneSampleCount / stats.PretrainSampleCount
            : float.MaxValue;

        if (imbalanceRatio > 10f)
            Tip("Критический дисбаланс данных. " +
                "Рассмотрите предобучение на большом открытом корпусе " +
                "(например, выгрузка из Википедии на нужном языке).");

        if (MaxSeqLen < 96)
            Tip($"MaxSeqLen={MaxSeqLen} обрезает длинные примеры. " +
                "При возможности увеличьте до 128+.");
    }
}

public class CorpusStats
{
    public int PretrainSampleCount { get; set; }
    public int FinetuneSampleCount { get; set; }
    public int TotalSampleCount => PretrainSampleCount + FinetuneSampleCount;

    public float AvgTokensPerSample { get; set; }
    public int MaxTokensInSample { get; set; }
    public int MinTokensInSample { get; set; }
    public int MedianTokensPerSample { get; set; }
    public int Percentile95Tokens { get; set; }

    public int UniqueWords { get; set; }
    public int UniqueChars { get; set; }
    public int TotalWords { get; set; }
    public float TypeTokenRatio { get; set; }

    public bool HasCyrillic { get; set; }
    public bool HasLatin { get; set; }
    public float CyrillicRatio { get; set; }
}

/// <summary>
/// Вычисляет оптимальные параметры LLMGPT2 из корпуса.
///
/// Ключевые правила выравнивания для gfx1150 (OpenCL, warp=32):
///   EmbeddingDim кратно 64
///   HiddenDim    = 4 × EmbeddingDim, кратно 128
///   headDim      = EmbeddingDim / NumHeads, кратно 32
///   MaxSeqLen    кратно 8
///   BpeVocabSize кратно 64
///
/// Особый случай: если finetune >> pretrain (chat-данные),
/// архитектура масштабируется по эффективному объёму данных,
/// а MaxSeqLen берётся из реального максимума корпуса.
/// </summary>
public class ModelConfigBuilder
{
    // ── Ограничения ──────────────────────────────────────────
    private const int MinEmbeddingDim = 64;
    private const int MaxEmbeddingDim = 512;
    private const int MinLayers = 1;
    private const int MaxLayers = 12;
    private const int MinMaxSeqLen = 64;
    private const int MaxMaxSeqLen = 512;
    private const int MinBpeVocabSize = 256;
    private const int MaxBpeVocabSize = 8192;

    // ═══════════════════════════════════════════════════════════
    // АНАЛИЗ КОРПУСА
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Анализирует тексты и возвращает статистику корпуса.
    /// maxSeqLenOverride — если вы уже знаете реальный максимум
    /// (например, из токенизатора), передайте его сюда.
    /// </summary>
    public static CorpusStats AnalyzeCorpus(
        List<string> pretrainData,
        List<string> finetuneData,
        int maxSeqLenOverride = 0)
    {
        var stats = new CorpusStats
        {
            PretrainSampleCount = pretrainData.Count,
            FinetuneSampleCount = finetuneData.Count,
        };

        var allTexts = pretrainData.Concat(finetuneData).ToList();
        if (allTexts.Count == 0) return stats;

        var allWords = new List<string>();
        var uniqueWords = new HashSet<string>();
        var uniqueChars = new HashSet<char>();
        var sampleWordCounts = new List<int>();
        int cyrillicChars = 0, latinChars = 0, totalChars = 0;

        foreach (var text in allTexts)
        {
            string clean = text
                .Replace("</s>", "").Replace("<bos>", "")
                .Replace("<user>", "").Replace("<assistant>", "")
                .Replace("<sep>", "").Trim();

            var words = clean.Split((char[])null!,
                StringSplitOptions.RemoveEmptyEntries);
            sampleWordCounts.Add(words.Length);

            foreach (var word in words)
            {
                allWords.Add(word);
                uniqueWords.Add(word.ToLowerInvariant());
                foreach (char c in word)
                {
                    uniqueChars.Add(c);
                    totalChars++;
                    if (c is >= 'а' and <= 'я' or >= 'А' and <= 'Я' or 'ё' or 'Ё')
                        cyrillicChars++;
                    else if (c is >= 'a' and <= 'z' or >= 'A' and <= 'Z')
                        latinChars++;
                }
            }
        }

        stats.UniqueWords = uniqueWords.Count;
        stats.UniqueChars = uniqueChars.Count;
        stats.TotalWords = allWords.Count;
        stats.TypeTokenRatio = allWords.Count > 0
            ? (float)uniqueWords.Count / allWords.Count : 0f;
        stats.HasCyrillic = cyrillicChars > 0;
        stats.HasLatin = latinChars > 0;
        stats.CyrillicRatio = totalChars > 0
            ? (float)cyrillicChars / totalChars : 0f;

        float bpeMult = cyrillicChars > latinChars ? 1.5f : 1.2f;
        var tokenEst = sampleWordCounts
            .Select(w => (int)(w * bpeMult))
            .OrderBy(x => x)
            .ToList();

        if (tokenEst.Count > 0)
        {
            stats.AvgTokensPerSample = (float)tokenEst.Average();
            stats.MinTokensInSample = tokenEst.First();
            stats.MaxTokensInSample = tokenEst.Last();
            stats.MedianTokensPerSample = tokenEst[tokenEst.Count / 2];
            stats.Percentile95Tokens = tokenEst[(int)(tokenEst.Count * 0.95)];
        }

        // Если передан реальный максимум из токенизатора — используем его
        if (maxSeqLenOverride > 0)
            stats.MaxTokensInSample = maxSeqLenOverride;

        return stats;
    }

    // ═══════════════════════════════════════════════════════════
    // ГЛАВНЫЙ МЕТОД
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Вычисляет оптимальную конфигурацию LLMGPT2 из корпуса.
    ///
    /// maxSeqLenOverride — реальный максимум токенов в корпусе,
    /// если вы его уже знаете (передайте из токенизатора).
    /// Если 0 — вычисляется из статистики по словам.
    /// </summary>
    public static ModelConfig BuildOptimalConfig(
        List<string> pretrainData,
        List<string> finetuneData,
        int maxSeqLenOverride = 0)
    {
        var stats = AnalyzeCorpus(pretrainData, finetuneData, maxSeqLenOverride);
        var cfg = new ModelConfig();

        ComputeArchitecture(cfg, stats);
        ComputeTraining(cfg, stats);
        cfg.ComputeEstimatedParameters();
        return cfg;
    }

    // ═══════════════════════════════════════════════════════════
    // ВЫЧИСЛЕНИЕ АРХИТЕКТУРЫ
    // ═══════════════════════════════════════════════════════════

    private static void ComputeArchitecture(ModelConfig cfg, CorpusStats stats)
    {
        // ── Эффективный объём данных ──────────────────────────────
        // При сильном дисбалансе pretrain << finetune
        // коэффициент finetune уменьшаем, чтобы не раздувать модель.
        // Логика: если pretrain < 50 примеров — finetune почти не влияет
        // на архитектуру, т.к. модель не имеет достаточной базы.
        float finetuneWeightForWidth = stats.PretrainSampleCount < 50 ? 0.05f :
                                       stats.PretrainSampleCount < 200 ? 0.15f : 0.3f;

        int effectiveSamplesForLayers = stats.PretrainSampleCount;
        int effectiveSamplesForWidth = stats.PretrainSampleCount
            + (int)(stats.FinetuneSampleCount * finetuneWeightForWidth);

        // ── MaxSeqLen ────────────────────────────────────────────
        // Берём реальный максимум из корпуса (MaxTokensInSample
        // уже содержит maxSeqLenOverride если он был передан).
        // +8 запас на спецтокены (BOS/EOS/SEP).
        int seqFromCorpus = stats.MaxTokensInSample > 0
            ? stats.MaxTokensInSample + 8
            : (int)(stats.Percentile95Tokens * 1.3f) + 8;

        cfg.MaxSeqLen = Math.Clamp(
            AlignTo(seqFromCorpus, 8),
            MinMaxSeqLen,
            MaxMaxSeqLen);

        // ── EmbeddingDim ─────────────────────────────────────────
        int embByVocab = stats.UniqueWords switch
        {
            < 200 => 64,
            < 500 => 128,
            < 2000 => 192,
            < 5000 => 256,
            _ => 320,
        };

        // Ограничение по объёму данных.
        // Слишком большая модель при малом датасете переобучается.
        int embCap = effectiveSamplesForWidth switch
        {
            < 100 => 64,
            < 300 => 128,
            < 600 => 192,
            < 1000 => 256,
            _ => MaxEmbeddingDim,
        };

        cfg.EmbeddingDim = AlignTo(
            Math.Clamp(
                Math.Min(embByVocab, embCap),
                MinEmbeddingDim,
                MaxEmbeddingDim),
            64);

        // ── HiddenDim = 4 × EmbeddingDim ─────────────────────────
        cfg.HiddenDim = AlignTo(cfg.EmbeddingDim * 4, 128);

        // ── NumHeads ─────────────────────────────────────────────
        cfg.NumHeads = BestNumHeads(cfg.EmbeddingDim);

        // ── NumLayers ────────────────────────────────────────────
        // Только pretrain определяет глубину.
        // Правило: 1 слой на каждые 100 pretrain примеров.
        int numLayers = Math.Max(1, effectiveSamplesForLayers / 100);
        cfg.NumLayers = Math.Clamp(numLayers, MinLayers, MaxLayers);

        // ── BpeVocabSize ─────────────────────────────────────────
        // Размер словаря ограничиваем по объёму pretrain данных:
        // при малом pretrain большой словарь модель не выучит.
        int maxVocabByPretrain = stats.PretrainSampleCount switch
        {
            < 50 => 512,
            < 200 => 1024,
            < 500 => 2048,
            < 2000 => 4096,
            _ => MaxBpeVocabSize,
        };

        float vocabMult = stats.CyrillicRatio > 0.3f ? 2.5f : 1.5f;
        int vocabRaw = (int)(stats.UniqueWords * vocabMult);
        cfg.BpeVocabSize = Math.Clamp(
            AlignTo(Math.Max(vocabRaw, MinBpeVocabSize), 64),
            MinBpeVocabSize,
            Math.Min(MaxBpeVocabSize, maxVocabByPretrain));
    }

    // ═══════════════════════════════════════════════════════════
    // ВЫЧИСЛЕНИЕ ПАРАМЕТРОВ ОБУЧЕНИЯ
    // ═══════════════════════════════════════════════════════════

    private static void ComputeTraining(ModelConfig cfg, CorpusStats stats)
    {
        int pretrainSamples = Math.Max(stats.PretrainSampleCount, 1);
        int finetuneSamples = Math.Max(stats.FinetuneSampleCount, 1);

        // ── Pretrain Epochs ───────────────────────────────────────
        // Целевое число шагов = 3000.
        // При 22 примерах: 3000/22 ≈ 136 эпох — адекватно.
        // При 1000 примерах: 3000/1000 = 3 → зажимаем до 10.
        const int targetPretrainSteps = 3000;
        cfg.PretrainEpochs = Math.Clamp(
            targetPretrainSteps / pretrainSamples,
            10, 500);

        // ── Finetune Epochs ───────────────────────────────────────
        // Целевое число шагов масштабируется от объёма данных.
        // Правило: не менее 3 полных прохода по датасету,
        // не более 50 эпох для больших датасетов.
        //
        // При 1452 примерах: минимум 3 прохода = 3 эпохи,
        // но для хорошей сходимости нужно 5-15 эпох.
        // Формула: целевые шаги = max(finetuneSamples*5, 5000)
        // гарантирует минимум 5 полных проходов.
        int targetFinetuneSteps = Math.Max(finetuneSamples * 5, 5000);
        cfg.FinetuneEpochs = Math.Clamp(
            targetFinetuneSteps / finetuneSamples,
            5, 50);

        // ── Learning Rate ─────────────────────────────────────────
        // Pretrain LR: обратно пропорционален EmbeddingDim.
        // EmbeddingDim=64  → ~3e-3
        // EmbeddingDim=128 → ~1.6e-3
        // EmbeddingDim=256 → ~7.8e-4
        cfg.PretrainLr = Math.Clamp(
            0.2f / cfg.EmbeddingDim,
            5e-4f, 3e-3f);

        // Finetune LR: 20% от pretrain LR
        cfg.FinetuneLr = Math.Clamp(
            cfg.PretrainLr * 0.2f,
            1e-4f, 5e-4f);

        // ── Warmup Steps ──────────────────────────────────────────
        // Warmup считается из реального числа шагов фазы.
        // Правило: 5-10% от total steps, но не менее разумного минимума.
        //
        // Pretrain: обычно малый датасет → warmup пропорционален эпохам.
        // Finetune: большой датасет → warmup должен покрыть ~первые эпохи.
        int totalPreSteps = cfg.PretrainEpochs * pretrainSamples;
        int totalFtSteps = cfg.FinetuneEpochs * finetuneSamples;

        // Pretrain warmup: 7% от total, но не более 300 шагов
        cfg.PretrainWarmupSteps = Math.Clamp(
            totalPreSteps / 14,
            10, 300);

        // Finetune warmup: 5% от total, но не более одной эпохи worth шагов.
        // Для 1452 примеров × 5 эпох = 7260 шагов → 5% = 363 шага ≈ 0.25 эпохи.
        int ftWarmupRaw = totalFtSteps / 20;
        // Ограничиваем: не больше чем 1 полная эпоха finetune
        cfg.FinetuneWarmupSteps = Math.Clamp(
            ftWarmupRaw,
            Math.Min(50, finetuneSamples),
            Math.Min(finetuneSamples, 500));

        // ── Gradient Clip ─────────────────────────────────────────
        // Малый датасет → нестабильные градиенты → жёстче клиппинг.
        int effectiveSamples = stats.PretrainSampleCount
            + (int)(stats.FinetuneSampleCount * 0.3f);
        cfg.GradientClipNorm = effectiveSamples < 100 ? 2.0f : 5.0f;
    }

    // ═══════════════════════════════════════════════════════════
    // ВСПОМОГАТЕЛЬНЫЕ
    // ═══════════════════════════════════════════════════════════

    /// <summary>Округляет v вверх до кратного alignment.</summary>
    private static int AlignTo(int v, int alignment)
        => ((v + alignment - 1) / alignment) * alignment;

    /// <summary>
    /// Возвращает максимальное число голов из {1,2,4,8,16}
    /// при условии headDim = embDim/heads кратно 32.
    /// </summary>
    private static int BestNumHeads(int embDim)
    {
        int best = 1;
        foreach (int h in new[] { 1, 2, 4, 8, 16 })
        {
            if (embDim % h != 0) continue;
            if ((embDim / h) % 32 != 0) continue;
            best = h;
        }
        return best;
    }


}