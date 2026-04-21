//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;

namespace LLM.ILGPU;

class Program
{
    // ── Пути по умолчанию ────────────────────────────────────
    const string DefaultPretrainData = "data/pretraining_data.json";
    const string DefaultChatData = "data/chat_training_data.json";

    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.InputEncoding = System.Text.Encoding.UTF8;

        using var gpuContext = new Context();

        ILLM? llm = null;
        ITokenizer? tokenizer = null;

    Main:
        Console.WriteLine("╔══════════════════════════════════════╗");
        Console.WriteLine("║     LLM ILGPU — Главное меню         ║");
        Console.WriteLine("╠══════════════════════════════════════╣");
        Console.WriteLine("║  1. Загрузить модель из файла        ║");
        Console.WriteLine("║  2. Создать новую GPT2 и обучить     ║");
        Console.WriteLine("║  3. Анализ корпуса                   ║");
        Console.WriteLine("╚══════════════════════════════════════╝");
        Console.Write("\nВыберите опцию (1, 2 или 3): ");

        var choice = Console.ReadLine()?.Trim();
        if (choice == "1") (llm, tokenizer) = TryLoadModel(gpuContext);
        else if (choice == "2") (llm, tokenizer) = CreateGPT2Minimal(gpuContext);
        else if (choice == "3") { AnalizeCorpus(); goto Main; }
        else goto Main;

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("╔══════════════════════════════════════╗");
            Console.WriteLine("║          Меню действий               ║");
            Console.WriteLine("╠══════════════════════════════════════╣");
            Console.WriteLine("║  1. Предсказать текст                ║");
            Console.WriteLine("║  2. Обучить модель                   ║");
            Console.WriteLine("║  3. Сохранить модель                 ║");
            Console.WriteLine("║  4. Интерактивный чат                ║");
            Console.WriteLine("║  5. Информация о модели              ║");
            Console.WriteLine("║  6. Тест токенизатора                ║");
            Console.WriteLine("║  7. Экспорт в GGUF                   ║");
            Console.WriteLine("║  8. Выход                            ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.Write("\nВыберите опцию: ");

            var action = Console.ReadLine()?.Trim();
            switch (action)
            {
                case "1": PredictText(llm); break;
                case "2": TrainModel(llm, tokenizer!); break;
                case "3": SaveModel(llm, tokenizer!); break;
                case "4": InteractiveChat(llm); break;
                case "5": ShowModelInfo(llm); break;
                case "6": TestTokenizer(tokenizer!); break;
                case "7": ExportToGgufV2(llm, tokenizer!); break;
                case "8":
                    Console.WriteLine("Выход из программы.");
                    llm?.Dispose();
                    return;
                default:
                    Console.WriteLine("Неверная опция.");
                    break;
            }
        }
    }

    private static void AnalizeCorpus()
    {

        // 1. Загружаем данные
        var dataset = new Dataset(
            ResolvePath(DefaultPretrainData),
            ResolvePath(DefaultChatData),
            DatasetType.JSON);

        var pretrainTexts = dataset.PretrainingData
            .Select(t => t.Replace("</s>", "").Trim())
            .ToList();

        var finetuneTexts = dataset.ChatTrainingData
            .Select(FormatChatForBpe)
            .Where(t => t != null)
            .Select(t => t!)
            .ToList();

        var allTexts = pretrainTexts.Concat(finetuneTexts).ToList();

        var configPreview = ModelConfigBuilder.BuildOptimalConfig(
            pretrainTexts, finetuneTexts);
        int vocabSize = configPreview.BpeVocabSize;

        // Обучаем токенизатор
        var tokenizer = new BPETokenizer();

        tokenizer.Train(
            corpus: allTexts,
            vocabSize: vocabSize,
            minFrequency: 1,
            verbose: false); //не отображаем процесс обучения

        // Вычисляем реальный MaxSeqLen из токенизированного корпуса
        int realMaxSeqLen = 0;

        for (int i = 0; i < allTexts.Count; i++)
        {
            var tokens = tokenizer.Encode(allTexts[i], addBos: true, addEos: true);
            if (tokens.Count > realMaxSeqLen)
            {
                realMaxSeqLen = tokens.Count;
            }
        }

        // 5. Строим финальный конфиг с реальным MaxSeqLen
        var config = ModelConfigBuilder.BuildOptimalConfig(
            pretrainTexts,
            finetuneTexts,
            maxSeqLenOverride: realMaxSeqLen);

        // выводим рекомендации
        var corpusStats = ModelConfigBuilder.AnalyzeCorpus(pretrainTexts, finetuneTexts, realMaxSeqLen);
        config.PrintRecommendations(corpusStats);
    }

    // ═══════════════════════════════════════════════════════════
    // ЗАГРУЗКА
    // ═══════════════════════════════════════════════════════════

    static (ILLM?, ITokenizer?) TryLoadModel(Context gpuContext)
    {
        Console.Write(
            "\nПуть к файлу модели (по умолчанию: models/model.bin): ");
        string? filePath = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(filePath))
            filePath = "models/model.bin";

        if (!Path.IsPathRooted(filePath))
            filePath = Path.Combine(AppContext.BaseDirectory, filePath);

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"Файл не найден: {filePath}");
            return (null, null);
        }

        try
        {
            Console.WriteLine($"Загрузка модели из {filePath}...");
            using var stream = File.Open(filePath, FileMode.Open);
            using var reader = new BinaryReader(stream);

            string marker = reader.ReadString();
            if (marker != "LLMv2_UNIFIED_V1")
                throw new InvalidDataException(
                    $"Неверный формат файла: {marker}");

            string archTag = reader.ReadString();
            int embDim = reader.ReadInt32();
            int hidDim = reader.ReadInt32();
            int nHeads = reader.ReadInt32();
            int maxSeq = reader.ReadInt32();

            long posBeforeToken = stream.Position;
            string tokMarker = reader.ReadString();

            ITokenizer tokenizer;
            if (tokMarker == "BPE2FIXED_V1")
            {
                stream.Position = posBeforeToken;
                tokenizer = BPETokenizer.LoadFromStream(reader);
            }
            else
            {
                throw new InvalidDataException(
                    $"Неизвестный формат токенизатора: '{tokMarker}'");
            }

            int layerCount = reader.ReadInt32();
            var layerTypes = new string[layerCount];
            for (int i = 0; i < layerCount; i++)
                layerTypes[i] = reader.ReadString();

            if (archTag != "GPT2")
            {
                Console.WriteLine(
                    $"⚠ Архитектура '{archTag}' не поддерживается.");
                return (null, null);
            }

            int numTransformerLayers =
                layerTypes.Count(t => t == "TransformerBlockPreNorm");

            var llm = new LLMGPT2(
                gpuContext, tokenizer,
                embeddingDim: embDim,
                hiddenDim: hidDim,
                numHeads: nHeads,
                numLayers: numTransformerLayers,
                maxSeqLen: maxSeq);

            var network = llm.GetLayers();
            if (network.Count != layerCount)
                throw new InvalidDataException(
                    $"Кол-во слоёв не совпадает: " +
                    $"файл={layerCount}, модель={network.Count}");

            for (int i = 0; i < layerCount; i++)
            {
                if (network[i].LayerType != layerTypes[i])
                    throw new InvalidDataException(
                        $"Слой {i}: тип в файле={layerTypes[i]}, " +
                        $"в модели={network[i].LayerType}");

                switch (network[i])
                {
                    case EmbeddingLayer emb:
                        LayerWeights.LoadEmbeddingGPUWeights(emb, reader);
                        break;
                    case PositionalEmbeddingLayer posEmb:
                        LayerWeights.LoadPositionalEmbeddingGPUWeights(posEmb, reader);
                        break;
                    case TransformerBlockPreNorm block:
                        LayerWeights.LoadPreNormBlockWeights(block, reader);
                        break;
                    case LayerNormLayer ln:
                        LayerWeights.LoadLayerNormWeights(ln, reader, "final_norm");
                        break;
                    case LinearLayer linear:
                        LayerWeights.LoadLinearWeights(linear, reader, "output");
                        break;
                    default:
                        Console.WriteLine(
                            $"  ⚠ Пропущен слой {i}: {layerTypes[i]}");
                        break;
                }
            }

            Console.WriteLine("  Прогрев GPU kernels...");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                llm.WarmUpInternal();
                gpuContext.Accelerator.Synchronize();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ⚠ WarmUp предупреждение: {ex.Message}");
            }
            sw.Stop();
            Console.WriteLine($"  Прогрев завершён за {sw.ElapsedMilliseconds}ms");
            Console.WriteLine("✓ Модель успешно загружена!");

            ShowModelInfo(llm);
            return (llm, tokenizer);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка загрузки: {ex.Message}");
            if (ex.InnerException != null)
                Console.WriteLine($"  → {ex.InnerException.Message}");
            return (null, null);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // СОХРАНЕНИЕ
    // ═══════════════════════════════════════════════════════════

    static void SaveModel(ILLM llm, ITokenizer tokenizer)
    {
        Console.Write(
            "\nПуть для сохранения (по умолчанию: models/model.bin): ");
        string? filePath = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(filePath))
            filePath = "models/model.bin";

        if (!Path.IsPathRooted(filePath))
            filePath = Path.Combine(AppContext.BaseDirectory, filePath);

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
            using var stream = File.Open(filePath, FileMode.Create);
            using var writer = new BinaryWriter(stream);

            writer.Write("LLMv2_UNIFIED_V1");

            string archTag = llm is LLMGPT2 ? "GPT2" : "V2";
            writer.Write(archTag);

            var layers = LayerWeights.GetNetworkLayers(llm);
            int embDim = 0, hidDim = 0, nHeads = 0, maxSeq = 0;

            foreach (var layer in layers)
            {
                switch (layer)
                {
                    case EmbeddingLayer emb:
                        embDim = emb.EmbeddingDim;
                        maxSeq = emb.MaxSeqLen;
                        break;
                    case TransformerBlockPreNorm block:
                        embDim = block.EmbeddingDim;
                        hidDim = block.HiddenDim;
                        nHeads = block.NumHeads;
                        break;
                }
            }

            if (embDim == 0)
                throw new InvalidOperationException(
                    "Не удалось определить конфигурацию модели");

            writer.Write(embDim);
            writer.Write(hidDim);
            writer.Write(nHeads);
            writer.Write(maxSeq);

            tokenizer.SaveToStream(writer);

            writer.Write(layers.Count);
            foreach (var layer in layers)
                writer.Write(layer.LayerType);

            foreach (var layer in layers)
            {
                switch (layer)
                {
                    case EmbeddingLayer emb:
                        LayerWeights.SaveEmbeddingGPUWeights(emb, writer);
                        break;
                    case PositionalEmbeddingLayer pos:
                        LayerWeights.SavePositionalEmbeddingGPUWeights(pos, writer);
                        break;
                    case LayerNormLayer ln:
                        LayerWeights.SaveLayerNormWeights(ln, writer, "final_norm");
                        break;
                    case TransformerBlockPreNorm block:
                        LayerWeights.SavePreNormBlockWeights(block, writer);
                        break;
                    case LinearLayer linear:
                        LayerWeights.SaveLinearWeights(linear, writer, "output");
                        break;
                    default:
                        Console.WriteLine(
                            $"  ⚠ Слой {layer.LayerType} не сохранён");
                        break;
                }
            }

            long sizeKB = new FileInfo(filePath).Length / 1024;
            Console.WriteLine($"✓ Модель сохранена: {filePath}");
            Console.WriteLine($"  Размер: {sizeKB:N0} KB");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка сохранения: {ex.Message}");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // СОЗДАНИЕ НОВОЙ МОДЕЛИ
    // ═══════════════════════════════════════════════════════════

    static (ILLM, ITokenizer) CreateGPT2Minimal(Context gpuContext)
    {
        Console.WriteLine("\n═══ СОЗДАНИЕ GPT2 ═══");

        // 1. Загружаем данные
        var dataset = new Dataset(
            ResolvePath(DefaultPretrainData),
            ResolvePath(DefaultChatData),
            DatasetType.JSON);

        var pretrainTexts = dataset.PretrainingData
            .Select(t => t.Replace("</s>", "").Trim())
            .ToList();

        var finetuneTexts = dataset.ChatTrainingData
            .Select(FormatChatForBpe)
            .Where(t => t != null)
            .Select(t => t!)
            .ToList();

        var allTexts = pretrainTexts.Concat(finetuneTexts).ToList();

        // 2. Определяем размер словаря из конфига (до обучения токенизатора)
        //    Используем предварительный анализ корпуса
        var statsPreview = ModelConfigBuilder.AnalyzeCorpus(
            pretrainTexts, finetuneTexts);

        // Размер vocab: из формулы конфига, но зажатый в разумные пределы
        // Токенизатор обучим на этом размере
        var configPreview = ModelConfigBuilder.BuildOptimalConfig(
            pretrainTexts, finetuneTexts);
        int vocabSize = configPreview.BpeVocabSize;


        Console.WriteLine(
            $"\nСтрок pretrain: {pretrainTexts.Count:N0}" +
            $"\nСтрок finetune: {finetuneTexts.Count:N0}");

        // 3. Обучаем токенизатор
        var tokenizer = new BPETokenizer();
        Console.WriteLine("\n═══ Обучение BpeTokenizer ═══");

        tokenizer.Train(
            corpus: allTexts,
            vocabSize: vocabSize,
            minFrequency: 1,
            verbose: true);

        Console.WriteLine($"  Vocab size: {tokenizer.VocabSize}");
        Console.WriteLine(
            $"  BOS={tokenizer.BosId}, EOS={tokenizer.EosId}, " +
            $"UNK={tokenizer.UnkId}, PAD={tokenizer.PadId}");

        // 4. Вычисляем реальный MaxSeqLen из токенизированного корпуса
        //    ПОСЛЕ обучения токенизатора — только так получим точные длины
        int realMaxSeqLen = 0;

        for (int i = 0; i < allTexts.Count; i++)
        {
            var tokens = tokenizer.Encode(allTexts[i], addBos: true, addEos: true);
            if (tokens.Count > realMaxSeqLen)
            {
                realMaxSeqLen = tokens.Count;
            }
        }

        // 5. Строим финальный конфиг с реальным MaxSeqLen
        var config = ModelConfigBuilder.BuildOptimalConfig(
            pretrainTexts,
            finetuneTexts,
            maxSeqLenOverride: realMaxSeqLen);

        // BpeVocabSize — берём из обученного токенизатора
        config.BpeVocabSize = tokenizer.VocabSize;

        // 6. Создаём модель
        var model = new LLMGPT2(
            gpuContext, tokenizer,
            embeddingDim: config.EmbeddingDim,
            hiddenDim: config.HiddenDim,
            numHeads: config.NumHeads,
            numLayers: config.NumLayers,
            maxSeqLen: config.MaxSeqLen);

        ShowModelInfo(model);

        // 7. Обучаем
        Console.Write("\nОбучить модель сейчас? (y/n, по умолчанию y): ");
        if (Console.ReadLine()?.Trim().ToLower() != "n")
        {
            TrainWithConfig2(model, tokenizer, config,
                pretrainTexts, finetuneTexts);
        }

        return (model, tokenizer);
    }

    // ═══════════════════════════════════════════════════════════
    // ОБУЧЕНИЕ — из ModelConfig
    // ═══════════════════════════════════════════════════════════

    static void TrainWithConfig(
        ILLM llm, ITokenizer tokenizer, ModelConfig config,
        List<string> pretrainTexts, List<string> finetuneTexts)
    {
        if (pretrainTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║   ФАЗА 1: ПРЕДВАРИТЕЛЬНОЕ ОБУЧЕНИЕ   ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {pretrainTexts.Count}\n");

            int epochs = ReadIntWithDefault(
                "Количество эпох pretrain", config.PretrainEpochs, 1, 10000);
            float lr = ReadFloatWithDefault(
                "Learning rate pretrain", config.PretrainLr);

            var tokenized = TokenizeDataset(
                pretrainTexts, tokenizer, config.MaxSeqLen);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized, epochs, lr, "Pretrain");
        }

        if (finetuneTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║      ФАЗА 2: INSTRUCTION TUNING      ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {finetuneTexts.Count}\n");

            int epochs = ReadIntWithDefault(
                "Количество эпох finetune", config.FinetuneEpochs, 0, 10000);
            float lr = ReadFloatWithDefault(
                "Learning rate finetune", config.FinetuneLr);

            var tokenized = TokenizeDataset(
                finetuneTexts, tokenizer, config.MaxSeqLen);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized, epochs, lr, "Finetune");
        }

        Console.WriteLine("\n═══ ТЕСТ ПОСЛЕ ОБУЧЕНИЯ ═══");
        Console.Write("Введите запрос: ");
        string? testInput = Console.ReadLine()?.Trim();
        if (!string.IsNullOrEmpty(testInput))
        {
            var result = llm.Predict(testInput);
            Console.WriteLine($"\nОтвет: {result}");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ОБУЧЕНИЕ — BPETokenizer
    // ═══════════════════════════════════════════════════════════

    static void TrainWithConfig2(
        ILLM llm, ITokenizer tokenizer, ModelConfig config,
        List<string> pretrainTexts, List<string> finetuneTexts)
    {
        int maxSeq = config.MaxSeqLen;

        var TokenizeWithBpe = (List<string> texts) => texts
            .Select(t => tokenizer.Encode(t, addBos: true, addEos: true))
            .Where(t => t.Count >= 2 && t.Count <= maxSeq)  // <= вместо <
            .ToList();

        if (pretrainTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║   ФАЗА 1: ПРЕДВАРИТЕЛЬНОЕ ОБУЧЕНИЕ  ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {pretrainTexts.Count}\n");

            int epochs = ReadIntWithDefault(
                "Количество эпох pretrain", config.PretrainEpochs, 1, 10000);
            float lr = ReadFloatWithDefault(
                "Learning rate pretrain", config.PretrainLr);

            var tokenized = TokenizeWithBpe(pretrainTexts);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized, epochs, lr, "Pretrain",
                    config.PretrainWarmupSteps);   // ← передаём warmup из конфига
        }

        if (finetuneTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║      ФАЗА 2: INSTRUCTION TUNING     ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {finetuneTexts.Count}\n");

            int epochs = ReadIntWithDefault(
                "Количество эпох finetune", config.FinetuneEpochs, 0, 10000);
            float lr = ReadFloatWithDefault(
                "Learning rate finetune", config.FinetuneLr);

            var tokenized = TokenizeWithBpe(finetuneTexts);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized, epochs, lr, "Finetune",
                    config.FinetuneWarmupSteps);   // ← передаём warmup из конфига
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ОБУЧЕНИЕ — планировщик LR (warmup + cosine decay)
    // ═══════════════════════════════════════════════════════════

    static void TrainPhaseWithSchedule(
        ILLM llm, List<List<int>> tokenizedData,
        int epochs, float baseLr, string phase,
        int warmupStepsOverride = 0)   // ← новый параметр из конфига
    {
        int totalSteps = epochs * tokenizedData.Count;

        // Если warmup передан из конфига — используем его.
        // Иначе fallback: 7% от total, но не более 300.
        int warmupSteps = warmupStepsOverride > 0
            ? warmupStepsOverride
            : Math.Clamp(totalSteps / 14, 10, 300);

        int globalStep = 0;

        Console.WriteLine($"  Warmup steps: {warmupSteps}");
        Console.WriteLine($"  Total steps:  {totalSteps}");
        Console.WriteLine($"  Base LR:      {baseLr}\n");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalLoss = 0f;
            int validCount = 0;

            var shuffled = tokenizedData
                .OrderBy(_ => Random.Shared.Next())
                .ToList();

            foreach (var tokens in shuffled)
            {
                if (tokens.Count < 2) continue;
                validCount++;
                globalStep++;

                float lr = ComputeLearningRate(baseLr, globalStep, warmupSteps, totalSteps);
                float loss = llm.TrainStep(tokens, lr);
                totalLoss += loss;
            }

            if (validCount > 0 &&
                (epoch % 5 == 0 || epoch == epochs - 1))
            {
                float avgLoss = totalLoss / validCount;
                float currentLr = ComputeLearningRate(baseLr, globalStep, warmupSteps, totalSteps);
                Console.WriteLine(
                    $"  [{phase}] Epoch {epoch,3}/{epochs}: " +
                    $"Loss = {avgLoss:F4}, LR = {currentLr:E2}");
            }
        }
    }

    static float ComputeLearningRate(
        float baseLr, int step, int warmupSteps, int totalSteps)
    {
        if (step < warmupSteps)
            return baseLr * step / Math.Max(warmupSteps, 1);

        float progress = (float)(step - warmupSteps)
                       / Math.Max(totalSteps - warmupSteps, 1);
        return baseLr * 0.5f * (1.0f + (float)Math.Cos(Math.PI * progress));
    }

    // ═══════════════════════════════════════════════════════════
    // ОБУЧЕНИЕ — из меню
    // ═══════════════════════════════════════════════════════════

    static void TrainModel(ILLM llm, ITokenizer tokenizer)
    {
        Console.WriteLine("\n═══ Обучение модели ═══");
        Console.WriteLine("1. Двухфазное обучение (pretrain + finetune)");
        Console.WriteLine("2. Только pretrain");
        Console.WriteLine("3. Только finetune (chat)");
        Console.WriteLine("4. Обучить на произвольном тексте");
        Console.Write("Выбор (1): ");

        var trainChoice = Console.ReadLine()?.Trim();

        try
        {
            var dataset = new Dataset(
                ResolvePath(DefaultPretrainData),
                ResolvePath(DefaultChatData),
                DatasetType.JSON);

            var pretrainTexts = dataset.PretrainingData
                .Select(t => tokenizer.FormatPretraining(
                    t.Replace("</s>", "").Trim()))
                .ToList();

            var finetuneTexts = dataset.ChatTrainingData
                .Select(FormatChatForBpe)
                .Where(t => t != null)
                .Select(t => t!)
                .ToList();

            // Вычисляем MaxSeqLen из модели (уже создана)
            int maxSeqLen = llm.MaxSeqLen;

            // Строим конфиг только для LR/Epochs defaults
            var config = ModelConfigBuilder.BuildOptimalConfig(
                pretrainTexts, finetuneTexts,
                maxSeqLenOverride: maxSeqLen);

            // MaxSeqLen берём из модели — не из конфига
            config.MaxSeqLen = maxSeqLen;

            switch (trainChoice)
            {
                case "2":
                    TrainSinglePhase(llm, tokenizer, pretrainTexts,
                        "Pretrain", config.PretrainEpochs,
                        config.PretrainLr, maxSeqLen);
                    break;
                case "3":
                    TrainSinglePhase(llm, tokenizer, finetuneTexts,
                        "Finetune", config.FinetuneEpochs,
                        config.FinetuneLr, maxSeqLen);
                    break;
                case "4":
                    TrainCustomText(llm, tokenizer, maxSeqLen);
                    break;
                default:
                    TrainWithConfig(llm, tokenizer, config,
                        pretrainTexts, finetuneTexts);
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка обучения: {ex.Message}");
        }
    }

    static void TrainSinglePhase(
        ILLM llm, ITokenizer tokenizer,
        List<string> texts, string phaseName,
        int defaultEpochs, float defaultLr, int maxSeqLen)
    {
        Console.WriteLine($"\n═══ {phaseName.ToUpper()} ═══");
        Console.WriteLine($"Примеров: {texts.Count}\n");

        int epochs = ReadIntWithDefault(
            "Количество эпох", defaultEpochs, 1, 10000);
        float lr = ReadFloatWithDefault("Learning rate", defaultLr);

        var tokenized = TokenizeDataset(texts, tokenizer, maxSeqLen);
        Console.WriteLine($"Валидных примеров: {tokenized.Count}");

        if (tokenized.Count > 0)
            TrainPhaseWithSchedule(llm, tokenized, epochs, lr, phaseName);
        else
            Console.WriteLine("Нет валидных примеров для обучения.");
    }

    static void TrainCustomText(
        ILLM llm, ITokenizer tokenizer, int maxSeqLen)
    {
        Console.WriteLine(
            "\nВведите тексты для обучения (пустая строка = конец):");
        var texts = new List<string>();

        while (true)
        {
            Console.Write("> ");
            string? line = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) break;
            texts.Add(tokenizer.FormatPretraining(line.Trim()));
        }

        if (texts.Count == 0)
        {
            Console.WriteLine("Нет данных для обучения.");
            return;
        }

        int epochs = ReadIntWithDefault("Количество эпох", 50, 1, 10000);
        float lr = ReadFloatWithDefault("Learning rate", 0.001f);

        var tokenized = TokenizeDataset(texts, tokenizer, maxSeqLen);
        Console.WriteLine($"Валидных примеров: {tokenized.Count}");

        if (tokenized.Count > 0)
            TrainPhaseWithSchedule(llm, tokenized, epochs, lr, "Custom");
    }

    // ═══════════════════════════════════════════════════════════
    // ПРЕДСКАЗАНИЕ
    // ═══════════════════════════════════════════════════════════

    static void PredictText(ILLM llm)
    {
        Console.Write("\nВведите текст: ");
        string? input = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.WriteLine("Пустой ввод.");
            return;
        }

        Console.Write("Temperature (0.7): ");
        float temp = ReadFloatWithDefault("Temperature", 0.7f);

        Console.WriteLine($"\nUser: {input}");
        string response = llm.Predict(input, temperature: temp);
        Console.WriteLine($"Assistant: {response}");
    }

    // ═══════════════════════════════════════════════════════════
    // ИНТЕРАКТИВНЫЙ ЧАТ
    // ═══════════════════════════════════════════════════════════

    static void InteractiveChat(ILLM llm)
    {
        Console.WriteLine("\n╔══════════════════════════════════════╗");
        Console.WriteLine("║         Интерактивный чат            ║");
        Console.WriteLine("╠══════════════════════════════════════╣");
        Console.WriteLine("║  exit   — выход                      ║");
        Console.WriteLine("║  temp N — установить temperature     ║");
        Console.WriteLine("║  clear  — очистить экран             ║");
        Console.WriteLine("╚══════════════════════════════════════╝");

        float temperature = 0.7f;

        while (true)
        {
            Console.Write($"\n[temp={temperature:F1}] You: ");
            string? userInput = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(userInput)) continue;

            string trimmed = userInput.Trim();

            if (trimmed.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Выход из чата.");
                break;
            }

            if (trimmed.Equals("clear", StringComparison.OrdinalIgnoreCase))
            {
                Console.Clear();
                continue;
            }

            if (trimmed.StartsWith("temp ", StringComparison.OrdinalIgnoreCase))
            {
                if (float.TryParse(trimmed[5..].Trim(), out float newTemp))
                {
                    temperature = Math.Clamp(newTemp, 0.01f, 2.0f);
                    Console.WriteLine(
                        $"  Temperature установлена: {temperature:F2}");
                }
                continue;
            }

            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                string response = llm.Predict(trimmed, temperature);
                sw.Stop();

                Console.WriteLine($"Assistant: {response}");
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  [{sw.ElapsedMilliseconds} ms]");
                Console.ResetColor();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Ошибка генерации: {ex.Message}");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ИНФОРМАЦИЯ О МОДЕЛИ
    // ═══════════════════════════════════════════════════════════

    static void ShowModelInfo(ILLM llm)
    {
        // Читаем параметры из слоёв модели — не из констант
        var layers = llm.GetLayers();
        int embDim = 0, hidDim = 0, nHeads = 0, nLayers = 0;

        foreach (var layer in layers)
        {
            switch (layer)
            {
                case EmbeddingLayer emb:
                    embDim = emb.EmbeddingDim;
                    break;
                case TransformerBlockPreNorm block:
                    hidDim = block.HiddenDim;
                    nHeads = block.NumHeads;
                    nLayers++;
                    break;
            }
        }

        Console.WriteLine("╔══════════════════════════════════════╗");
        Console.WriteLine("║       ИНФОРМАЦИЯ О МОДЕЛИ            ║");
        Console.WriteLine("╠══════════════════════════════════════╣");
        Console.WriteLine($"║  Архитектура: {llm.NetworkDescription()}");
        Console.WriteLine($"║  Vocab size:  {llm.Tokenizer.VocabSize}");
        Console.WriteLine($"║  Параметры:   {llm.TotalParameters():N0}");
        Console.WriteLine($"║  Emb dim:     {embDim}");
        Console.WriteLine($"║  Hidden dim:  {hidDim}");
        Console.WriteLine($"║  Num heads:   {nHeads}");
        Console.WriteLine($"║  Num layers:  {nLayers}");
        Console.WriteLine($"║  Max seq len: {llm.MaxSeqLen}");
        Console.WriteLine($"║  Параметры:   {llm.TotalParameters():N0}");
        Console.WriteLine("╚══════════════════════════════════════╝");
    }

    // ═══════════════════════════════════════════════════════════
    // ЭКСПОРТ
    // ═══════════════════════════════════════════════════════════

    static void ExportToGgufV2(ILLM llm, ITokenizer tokenizer)
    {
        Console.Write("\nНазвание модели (по умолчанию: llm-ilgpu-v2): ");
        string? name = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(name)) name = "llm-ilgpu-v2";

        Console.Write("Путь для экспорта (по умолчанию: models/model.gguf): ");
        string? path = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(path)) path = "models/model.gguf";

        if (!Path.IsPathRooted(path))
            path = Path.Combine(AppContext.BaseDirectory, path);

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            if (llm is LLMGPT2 gpt2Model)
                GgufExporter.Export(gpt2Model, tokenizer, path, name);
            else
                Console.WriteLine(
                    $"⚠ Экспорт не поддерживается для {llm.GetType().Name}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка экспорта: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ТЕСТ ТОКЕНИЗАТОРА
    // ═══════════════════════════════════════════════════════════

    static void TestTokenizer(ITokenizer tokenizer)
    {
        Console.WriteLine("\n═══ Тест BPE токенизатора ═══");
        Console.WriteLine("Введите текст (пустая строка = выход):");

        while (true)
        {
            Console.Write("\n> ");
            string? input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input)) break;

            var ids = tokenizer.Encode(input, addBos: false, addEos: false);
            string decoded = tokenizer.Decode(ids);

            Console.WriteLine(
                $"  Token IDs [{ids.Count}]: [{string.Join(", ", ids)}]");

            var subwords = ids
                .Select(id => tokenizer.ToVocab().DecodeToken(id) ?? "?")
                .ToList();
            Console.WriteLine(
                $"  Подслова: [{string.Join("|", subwords)}]");
            Console.WriteLine($"  Decoded: \"{decoded}\"");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // УТИЛИТЫ
    // ═══════════════════════════════════════════════════════════

    static string? FormatChatForBpe(string text)
    {
        string clean = text.Replace("</s>", "").Trim();

        int userIdx = clean.IndexOf(
            "User:", StringComparison.OrdinalIgnoreCase);
        int assistantIdx = clean.IndexOf(
            "Assistant:", StringComparison.OrdinalIgnoreCase);

        if (userIdx >= 0 && assistantIdx > userIdx)
        {
            string userText = clean[(userIdx + 5)..assistantIdx]
                .Trim().TrimEnd(':').Trim();
            string assistantText = clean[(assistantIdx + 10)..]
                .Trim().TrimEnd(':').Trim();

            if (!string.IsNullOrEmpty(userText) &&
                !string.IsNullOrEmpty(assistantText))
                return $"<user> {userText}<assistant> {assistantText}";
        }

        if (!string.IsNullOrEmpty(clean))
            return clean.Trim();

        return null;
    }

    static List<List<int>> TokenizeDataset(
        List<string> texts, ITokenizer tokenizer, int maxSeqLen)
    {
        return texts
            .Select(t => tokenizer.Encode(t, addBos: true, addEos: true))
            .Where(ids => ids.Count >= 2 && ids.Count < maxSeqLen)
            .ToList();
    }

    static int ReadIntWithDefault(
        string prompt, int defaultValue,
        int min = 0, int max = int.MaxValue)
    {
        Console.Write($"  {prompt} (по умолчанию {defaultValue}): ");
        string? input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input)) return defaultValue;
        if (int.TryParse(input, out int value))
            return Math.Clamp(value, min, max);
        Console.WriteLine($"  Некорректный ввод, используется {defaultValue}");
        return defaultValue;
    }

    static float ReadFloatWithDefault(string prompt, float defaultValue)
    {
        Console.Write($"  {prompt} (по умолчанию {defaultValue:G4}): ");
        string? input = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(input)) return defaultValue;

        if (float.TryParse(input,
            System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture,
            out float value))
            return value;

        if (float.TryParse(input, out value))
            return value;

        Console.WriteLine(
            $"  Некорректный ввод, используется {defaultValue:G4}");
        return defaultValue;
    }

    static string ResolvePath(string relativePath)
    {
        if (Path.IsPathRooted(relativePath))
            return relativePath;
        return Path.Combine(AppContext.BaseDirectory, relativePath);
    }
}