//https://github.com/virex-84

using ILGPU;
using ILGPU.Runtime;

namespace LLM.ILGPU;

class Program
{
    // ═══════════════════════════════════════════════════════════
    // Конфигурация модели
    // ═══════════════════════════════════════════════════════════
    const int EmbeddingDim = 64;
    const int HiddenDim = 128;
    const int NumHeads = 2;
    const int NumLayers = 1;
    const int VocabSize = 512;
    const int MaxSeqLen = 80;

    // Пути по умолчанию
    const string DefaultPretrainData = "data/pretraining_data.json";
    const string DefaultChatData = "data/chat_training_data.json";

    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.InputEncoding = System.Text.Encoding.UTF8;

        using var gpuContext = new Context();

        ILLM? llm = null;
        ITokenizer? tokenizer = null;

    // ═══ Главное меню ═══
    Main:
        Console.WriteLine("╔══════════════════════════════════════╗");
        Console.WriteLine("║   LLM ILGPU — Главное меню           ║");
        Console.WriteLine("╠══════════════════════════════════════╣");
        Console.WriteLine("║ 1. Загрузить модель из файла         ║");
        Console.WriteLine("║ 2. Создать новую GPT2 и обучить      ║");
        Console.WriteLine("╚══════════════════════════════════════╝");
        Console.Write("\nВыберите опцию (1 или 2): ");

        var choice = Console.ReadLine()?.Trim();

        if (choice == "1")
        {
            (llm, tokenizer) = TryLoadModel(gpuContext);
        }
        else if (choice == "2")
        {
            (llm, tokenizer) = CreateGPT2Minimal(gpuContext);
        }
        else goto Main;

            // ═══ Меню действий ═══
            while (true)
            {
                Console.WriteLine();
                Console.WriteLine("╔══════════════════════════════════════╗");
                Console.WriteLine("║          Меню действий               ║");
                Console.WriteLine("╠══════════════════════════════════════╣");
                Console.WriteLine("║ 1. Предсказать текст                 ║");
                Console.WriteLine("║ 2. Обучить модель                    ║");
                Console.WriteLine("║ 3. Сохранить модель                  ║");
                Console.WriteLine("║ 4. Интерактивный чат                 ║");
                Console.WriteLine("║ 5. Информация о модели               ║");
                Console.WriteLine("║ 6. Тест токенизатора                 ║");
                Console.WriteLine("║ 7. Экспорт в GGUF                    ║");
                Console.WriteLine("║ 8. Выход                             ║");
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
                        llm.Dispose();
                        return;
                    default:
                        Console.WriteLine("Неверная опция.");
                        break;
                }
            }
    }

    // ═══════════════════════════════════════════════════════════
    // ЗАГРУЗКА МОДЕЛИ
    // ═══════════════════════════════════════════════════════════

    static (ILLM?, ITokenizer?) TryLoadModel(Context gpuContext)
    {
        Console.Write($"\nПуть к файлу модели (по умолчанию: models/model.bin): ");
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

            // ─── Заголовок ───
            string marker = reader.ReadString();
            if (marker != "LLMv2_UNIFIED_V1")
                throw new InvalidDataException(
                    $"Неверный формат файла: {marker}");

            // ─── Архитектура (новый формат с тегом) ───
            string archTag = "V2";
            int embDim, hidDim, nHeads, maxSeq;
            
            // Проверяем есть ли тег архитектуры
            long posAfterMarker = stream.Position;
            string possibleTag = reader.ReadString();

            archTag = possibleTag;
            embDim = reader.ReadInt32();
            hidDim = reader.ReadInt32();
            nHeads = reader.ReadInt32();
            maxSeq = reader.ReadInt32();

            Console.WriteLine($"  Architecture: {archTag}");
            Console.WriteLine($"  Config: emb={embDim}, hid={hidDim}, " +
                              $"heads={nHeads}, maxSeq={maxSeq}");

            // ─── Токенизатор (из модели!) ───
            long posBeforeToken = stream.Position;
            string markerCheck = reader.ReadString();

            ITokenizer tokenizer;

            if (markerCheck == "BPE2FIXED_V1")
            {
                // LoadFromStream ожидает маркер первым — читаем с позиции ДО маркера
                stream.Position = posBeforeToken;
                tokenizer = BPETokenizer.LoadFromStream(reader);
                Console.WriteLine($"  BpeTokenizer (GPT-2 byte-level)");
            }
            else
            {
                throw new InvalidDataException($"Неизвестный формат токенизатора: '{markerCheck}'");
            }
           Console.WriteLine($"  Vocab size: {tokenizer.VocabSize}");

            // ─── Архитектура ───
            int layerCount = reader.ReadInt32();
            var layerTypes = new string[layerCount];
            for (int i = 0; i < layerCount; i++)
                layerTypes[i] = reader.ReadString();

            Console.WriteLine($"  Layers: {string.Join(" → ", layerTypes)}");

            // ─── Воссоздаём слои и загружаем веса ───
            var acc = gpuContext.Accelerator;
            var layers = new List<ILayer>();

            for (int i = 0; i < layerCount; i++)
            {
                switch (layerTypes[i])
                {
                    case "Embedding":
                        var emb = new EmbeddingLayer(acc,
                            tokenizer.VocabSize, embDim, maxSeq);
                        LayerWeights.LoadEmbeddingGPUWeights(emb, reader);
                        layers.Add(emb);
                        break;

                    case "PositionalEmbedding":
                        var posEmb = new PositionalEmbeddingLayer(acc, maxSeq, embDim);
                        LayerWeights.LoadPositionalEmbeddingGPUWeights(posEmb, reader);
                        layers.Add(posEmb);
                        break;

                    case "TransformerBlockPreNorm":
                        var preNormBlock = new TransformerBlockPreNorm(acc,
                            embDim, nHeads, hidDim, maxSeq);
                        LayerWeights.LoadPreNormBlockWeights(preNormBlock, reader);
                        layers.Add(preNormBlock);
                        break;

                    case "LayerNorm":
                        var ln = new LayerNormLayer(acc, embDim, maxSeq);
                        LayerWeights.LoadLayerNormWeights(ln, reader, "final_norm");
                        layers.Add(ln);
                        break;

                    case "Linear":
                        var linear = new LinearLayer(acc, embDim,
                            tokenizer.VocabSize, true, maxSeq);
                        LayerWeights.LoadLinearWeights(
                            linear, reader, "output");
                        layers.Add(linear);
                        break;

                    default:
                        Console.WriteLine($"  ⚠ Пропущен неизвестный слой: {layerTypes[i]}");
                        break;
                }
            }

            // Создаём модель
            ILLM llm;
            if (archTag == "GPT2")
            {
                llm = new LLMGPT2(gpuContext, tokenizer,
                    embeddingDim: embDim,
                    hiddenDim: hidDim,
                    numHeads: nHeads,
                    numLayers: layerTypes.Count(t => t == "TransformerBlockPreNorm"),
                    maxSeqLen: maxSeq);

                // Заменяем слои в LLMGPT2 на загруженные
                var gpt2Net = LayerWeights.GetField<List<ILayer>>(llm, "_network");
                gpt2Net.Clear();
                foreach (var l in layers) gpt2Net.Add(l);

                Console.WriteLine($"  Модель: LLMGPT2");
            }
            else
            {
                Console.WriteLine($"⚠ Архитектура '{archTag}' не поддерживается. Используйте LLMGPT2.");
                return (null, null);
            }
            
            Console.WriteLine("✓ Модель успешно загружена!");
            ShowModelInfo(llm);

            return (llm, tokenizer);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка загрузки: {ex.Message}");
            return (null, null);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // СОХРАНЕНИЕ
    // ═══════════════════════════════════════════════════════════
    static void SaveModel(ILLM llm, ITokenizer tokenizer)
    {
        Console.Write($"\nПуть для сохранения (по умолчанию: models/model.bin): ");
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

            // ─── Заголовок ───
            writer.Write("LLMv2_UNIFIED_V1");

            // Определяем архитектуру
            string archTag = llm is LLMGPT2 ? "GPT2" : "V2";
            writer.Write(archTag);

            // ─── Конфигурация архитектуры ───
            // Для GPT2 записываем её параметры
            var gptLayers = LayerWeights.GetNetworkLayers(llm);
            int embDim = 0, hidDim = 0, nHeads = 0, nLayers = 0, maxSeq = 0;
            foreach (var layer in gptLayers)
            {
                if (layer is EmbeddingLayer emb)
                {
                    embDim = emb.EmbeddingDim;
                    maxSeq = emb.MaxSeqLen;
                }
                else if (layer is TransformerBlockPreNorm block)
                {
                    nLayers++;
                    embDim = block.EmbeddingDim;
                    hidDim = block.HiddenDim;
                    nHeads = block.NumHeads;
                }
            }
            writer.Write(embDim);
            writer.Write(hidDim);
            writer.Write(nHeads);
            writer.Write(maxSeq);

            // ─── Токенизатор (внутри модели!) ───
            tokenizer.SaveToStream(writer);

            // ─── Описание сети ───
            var layers = LayerWeights.GetNetworkLayers(llm);
            writer.Write(layers.Count);

            // ─── Типы слоёв (для воссоздания архитектуры при загрузке) ───
            foreach (var layer in layers)
                writer.Write(layer.LayerType);

            // ─── Веса слоёв ───
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
                }
            }

            Console.WriteLine($"✓ Модель сохранена: {filePath}");
            Console.WriteLine($"  Размер файла: {new FileInfo(filePath).Length / 1024} KB");
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
        Console.WriteLine("\n═══ СОЗДАНИЕ МИНИМАЛЬНОЙ GPT2 ═══");
        Console.WriteLine("   Архитектура: GPT-2 (Pre-Norm, GELU, обучаемые positional)");

        // 1. Загружаем данные
        string pretrainPath = ResolvePath(DefaultPretrainData);
        string chatPath = ResolvePath(DefaultChatData);
        var dataset = new Dataset(pretrainPath, chatPath, DatasetType.JSON);

        var pretrainTexts = dataset.PretrainingData
            .Select(t => t.Replace("</s>", "").Trim())
            .ToList();

        var finetuneTexts = dataset.ChatTrainingData
            .Select(FormatChatForBpe)
            .Where(t => t != null).Select(t => t!).ToList();

        var allTexts = pretrainTexts.Concat(finetuneTexts).ToList();

        // 3. Создаём/обучаем токенизатор — только BpeTokenizer
        ITokenizer tokenizer2;
        // BpeTokenizer — обучаемый через Train()
        Console.WriteLine("\n═══ Обучение BpeTokenizer ═══");
        var bpe3 = new BPETokenizer();

        bpe3.Train(
            corpus: allTexts,
            vocabSize: VocabSize,
            minFrequency: 1,
            verbose: true);

        Console.WriteLine($"  Vocab size: {bpe3.VocabSize}");
        Console.WriteLine($"  BOS={bpe3.BosId}, EOS={bpe3.EosId}, UNK={bpe3.UnkId}, PAD={bpe3.PadId}");
        tokenizer2 = bpe3;

        // 4. Вычисляем MaxSeqLen из корпуса (НЕ хардкодим!)
        int maxTokLen = 0;
        int maxLineIdx = 0;
        for (int i = 0; i < allTexts.Count; i++)
        {
            var ids = tokenizer2.Encode(allTexts[i], addBos:true, addEos:true);
            if (ids.Count > maxTokLen)
            {
                maxTokLen = ids.Count;
                maxLineIdx = i;
            }
        }

        int MaxSeqLen = maxTokLen;
        Console.WriteLine($"\n📊 MaxSeqLen из корпуса: {MaxSeqLen} (строка {maxLineIdx})");

        if (MaxSeqLen < 10)
        {
            Console.WriteLine($"\n⚠ MaxSeqLen слишком мал, ставим минимум 10");
            MaxSeqLen = 10;
        }
        else if (MaxSeqLen > 2048)
        {
            Console.WriteLine($"\n⚠ MaxSeqLen > 2048, ограничиваем 2048");
            MaxSeqLen = 2048;
        }

        // 5. Создаём модель
        var model = new LLMGPT2(
            gpuContext, tokenizer2,
            embeddingDim: EmbeddingDim,
            hiddenDim: HiddenDim,
            numHeads: NumHeads,
            numLayers: NumLayers,
            maxSeqLen: MaxSeqLen);

        // 4. Конфигурация обучения
        var config = new ModelConfig
        {
            EmbeddingDim = EmbeddingDim,
            HiddenDim = HiddenDim,
            NumHeads = NumHeads,
            NumLayers = NumLayers,
            MaxSeqLen = MaxSeqLen,
            BpeVocabSize = VocabSize,
            PretrainEpochs = 30,
            PretrainLr = 1e-3f,
            PretrainWarmupSteps = 50,
            FinetuneEpochs = 10,
            FinetuneLr = 1e-4f,
            FinetuneWarmupSteps = 20,
            GradientClipNorm = 5.0f,
            Profile = ModelProfile.Small
        };

        Console.WriteLine($"\n  Параметры модели: {model.TotalParameters():N0}");
        Console.WriteLine($"  Архитектура: {model.NetworkDescription()}");

        // 5. Обучаем
        Console.Write("\nОбучить модель сейчас? (y/n, по умолчанию y): ");
        if (Console.ReadLine()?.Trim().ToLower() != "n")
        {
            TrainWithConfig2(model, tokenizer2, config,
                pretrainTexts, finetuneTexts);
        }

        return (model, tokenizer2);
    }

    /// <summary>
    /// Обучение с параметрами из ModelConfig.
    /// </summary>
    static void TrainWithConfig(ILLM llm, ITokenizer tokenizer,
        ModelConfig config, List<string> pretrainTexts, List<string> finetuneTexts)
    {
        // ── Фаза 1: Pretrain ──
        if (pretrainTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║   ФАЗА 1: ПРЕДВАРИТЕЛЬНОЕ ОБУЧЕНИЕ   ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {pretrainTexts.Count}\n");

            int pretrainEpochs = ReadIntWithDefault(
                "Количество эпох pretrain", config.PretrainEpochs, 1, 10000);

            float pretrainLr = ReadFloatWithDefault(
                "Learning rate pretrain", config.PretrainLr);

            var tokenized = TokenizeDataset(pretrainTexts, tokenizer, config.MaxSeqLen);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized,
                    pretrainEpochs, pretrainLr, "Pretrain");
        }

        // ── Фаза 2: Finetune ──
        if (finetuneTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║   ФАЗА 2: INSTRUCTION TUNING         ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {finetuneTexts.Count}\n");

            int finetuneEpochs = ReadIntWithDefault("Количество эпох finetune", config.FinetuneEpochs, 0, 10000);

            float finetuneLr = ReadFloatWithDefault("Learning rate finetune", config.FinetuneLr);

            var tokenized = TokenizeDataset(finetuneTexts, tokenizer, config.MaxSeqLen);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized,
                    finetuneEpochs, finetuneLr, "Finetune");
        }

        // ── Тест ──
        Console.WriteLine("\n═══ ТЕСТ ПОСЛЕ ОБУЧЕНИЯ ═══");
        Console.Write("Введите запрос: ");
        string? testInput = Console.ReadLine()?.Trim();
        if (!string.IsNullOrEmpty(testInput))
        {
            var result = llm.Predict(testInput);
            Console.WriteLine($"\nОтвет: {result}");
        }
    }

    /// <summary>
    /// Обучение с BpeTokenizer (GPT-2 byte-level).
    /// </summary>
    static void TrainWithConfig2(ILLM llm, ITokenizer tokenizer,
        ModelConfig config, List<string> pretrainTexts, List<string> finetuneTexts)
    {
        // Токенизация BpeTokenizer — фильтруем по config.MaxSeqLen!
        int maxSeq = config.MaxSeqLen;
        var TokenizeWithBpe2 = (List<string> texts) => texts
            .Select(t => tokenizer.Encode(t, addBos:true, addEos:true))
            .Where(t => t.Count >= 2 && t.Count < maxSeq)
            .ToList();

        // ── Фаза 1: Pretrain ──
        if (pretrainTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║   ФАЗА 1: ПРЕДВАРИТЕЛЬНОЕ ОБУЧЕНИЕ   ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {pretrainTexts.Count}\n");

            int pretrainEpochs = ReadIntWithDefault("Количество эпох pretrain", config.PretrainEpochs, 1, 10000);

            float pretrainLr = ReadFloatWithDefault("Learning rate pretrain", config.PretrainLr);

            var tokenized = TokenizeWithBpe2(pretrainTexts);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized, pretrainEpochs, pretrainLr, "Pretrain");
        }

        // ── Фаза 2: Finetune ──
        if (finetuneTexts.Count > 0)
        {
            Console.WriteLine("\n╔══════════════════════════════════════╗");
            Console.WriteLine("║   ФАЗА 2: INSTRUCTION TUNING         ║");
            Console.WriteLine("╚══════════════════════════════════════╝");
            Console.WriteLine($"Примеров: {finetuneTexts.Count}\n");

            int finetuneEpochs = ReadIntWithDefault("Количество эпох finetune", config.FinetuneEpochs, 0, 10000);

            float finetuneLr = ReadFloatWithDefault("Learning rate finetune", config.FinetuneLr);

            var tokenized = TokenizeWithBpe2(finetuneTexts);
            Console.WriteLine($"Валидных примеров: {tokenized.Count}");

            if (tokenized.Count > 0)
                TrainPhaseWithSchedule(llm, tokenized, finetuneEpochs, finetuneLr, "Finetune");
        }
    }

    /// <summary>
    /// Обучение одной фазы с warmup + cosine decay.
    /// </summary>
    static void TrainPhaseWithSchedule(ILLM llm,
        List<List<int>> tokenizedData, int epochs, float baseLr, string phase)
    {
        int totalSteps = epochs * tokenizedData.Count;
        int warmupSteps = Math.Min(totalSteps / 10, 200);
        int globalStep = 0;

        Console.WriteLine($"  Warmup steps: {warmupSteps}");
        Console.WriteLine($"  Total steps: {totalSteps}");
        Console.WriteLine($"  Base LR: {baseLr}\n");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalLoss = 0;
            int validCount = 0;

            // Перемешиваем каждую эпоху
            var shuffled = tokenizedData
                .OrderBy(_ => Random.Shared.Next()).ToList();

            foreach (var tokens in shuffled)
            {
                if (tokens.Count < 2) continue;
                validCount++;
                globalStep++;

                float lr = ComputeLearningRate(baseLr, globalStep,
                    warmupSteps, totalSteps);

                float loss = llm.TrainStep(tokens, lr);
                totalLoss += loss;
            }

            if (validCount > 0 && (epoch % 5 == 0 || epoch == epochs - 1))
            {
                float avgLoss = totalLoss / validCount;
                float currentLr = ComputeLearningRate(baseLr, globalStep,
                    warmupSteps, totalSteps);
                Console.WriteLine(
                    $"  [{phase}] Epoch {epoch,3}/{epochs}: " +
                    $"Loss = {avgLoss:F4}, LR = {currentLr:E2}");
            }
        }
    }

    static float ComputeLearningRate(float baseLr, int step,
        int warmupSteps, int totalSteps)
    {
        if (step < warmupSteps)
            return baseLr * step / Math.Max(warmupSteps, 1);

        float progress = (float)(step - warmupSteps) / Math.Max(totalSteps - warmupSteps, 1);

        return baseLr * 0.5f * (1.0f + (float)Math.Cos(Math.PI * progress));
    }

    // ═══════════════════════════════════════════════════════════
    // ОБУЧЕНИЕ (из меню)
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

        string pretrainPath = ResolvePath(DefaultPretrainData);
        string chatPath = ResolvePath(DefaultChatData);

        try
        {
            var dataset = new Dataset(pretrainPath, chatPath, DatasetType.JSON);

            var pretrainTexts = dataset.PretrainingData
                .Select(t => tokenizer.FormatPretraining(
                    t.Replace("</s>", "").Trim()
                    ))
                .ToList();

            var finetuneTexts = dataset.ChatTrainingData
                .Select(FormatChatForBpe)
                .Where(t => t != null).Select(t => t!).ToList();

            // Анализируем корпус для defaults
            var stats = ModelConfigBuilder.AnalyzeCorpus(pretrainTexts, finetuneTexts);

            var config = ModelConfigBuilder.CreateConfig(ModelProfile.Auto, stats);

            switch (trainChoice)
            {
                case "2":
                    TrainSinglePhase(llm, tokenizer, pretrainTexts, "Pretrain", config.PretrainEpochs, config.PretrainLr,
                        config.MaxSeqLen);
                    break;
                case "3":
                    TrainSinglePhase(llm, tokenizer, finetuneTexts, "Finetune", config.FinetuneEpochs, config.FinetuneLr,
                        config.MaxSeqLen);
                    break;
                case "4":
                    TrainCustomText(llm, tokenizer, config.MaxSeqLen);
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

    /// <summary>
    /// Обучение одной фазы с интерактивным вводом параметров.
    /// </summary>
    static void TrainSinglePhase(ILLM llm, ITokenizer tokenizer,
        List<string> texts, string phaseName,
        int defaultEpochs, float defaultLr, int maxSeqLen)
    {
        Console.WriteLine($"\n═══ {phaseName.ToUpper()} ═══");
        Console.WriteLine($"Примеров: {texts.Count}\n");

        int epochs = ReadIntWithDefault("Количество эпох", defaultEpochs, 1, 10000);

        float lr = ReadFloatWithDefault("Learning rate", defaultLr);

        var tokenized = TokenizeDataset(texts, tokenizer, maxSeqLen);
        Console.WriteLine($"Валидных примеров: {tokenized.Count}");

        if (tokenized.Count > 0)
            TrainPhaseWithSchedule(llm, tokenized, epochs, lr, phaseName);
        else
            Console.WriteLine("Нет валидных примеров для обучения.");
    }

    static void TrainCustomText(ILLM llm, ITokenizer tokenizer, int maxSeqLen)
    {
        Console.WriteLine("\nВведите тексты для обучения (пустая строка = конец):");
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
    // Утилиты ввода с defaults
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Читает int с показом значения по умолчанию. Enter = default.
    /// </summary>
    static int ReadIntWithDefault(string prompt, int defaultValue,
        int min = 0, int max = int.MaxValue)
    {
        Console.Write($"  {prompt} (по умолчанию {defaultValue}): ");
        string? input = Console.ReadLine()?.Trim();

        if (string.IsNullOrEmpty(input))
            return defaultValue;

        if (int.TryParse(input, out int value))
            return Math.Clamp(value, min, max);

        Console.WriteLine($"    Некорректный ввод, используется {defaultValue}");
        return defaultValue;
    }

    /// <summary>
    /// Читает float с показом значения по умолчанию. Enter = default.
    /// </summary>
    static float ReadFloatWithDefault(string prompt, float defaultValue)
    {
        Console.Write($"  {prompt} (по умолчанию {defaultValue:G4}): ");
        string? input = Console.ReadLine()?.Trim();

        if (string.IsNullOrEmpty(input))
            return defaultValue;

        if (float.TryParse(input, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float value))
            return value;

        // Попробуем с запятой
        if (float.TryParse(input, out value))
            return value;

        Console.WriteLine($"    Некорректный ввод, используется {defaultValue:G4}");
        return defaultValue;
    }

    static float ReadFloat(float defaultValue) =>
    ReadFloatWithDefault("Значение", defaultValue);

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
        float temp = ReadFloat(0.7f);

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
        Console.WriteLine("║       Интерактивный чат               ║");
        Console.WriteLine("╠══════════════════════════════════════╣");
        Console.WriteLine("║  Команды:                            ║");
        Console.WriteLine("║    exit     — выход                  ║");
        Console.WriteLine("║    temp N   — установить temperature ║");
        Console.WriteLine("║    clear    — очистить экран         ║");
        Console.WriteLine("╚══════════════════════════════════════╝");

        float temperature = 0.7f;

        while (true)
        {
            Console.Write($"\n[temp={temperature:F1}] You: ");
            string? userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput)) continue;

            string trimmed = userInput.Trim();

            // Команды
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
                    Console.WriteLine($"  Temperature установлена: {temperature:F2}");
                }
                continue;
            }

            // Генерация ответа
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
    // Метод экспорта
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
            // LLMGPT2 → специальный экспортёр
            if (llm is LLMGPT2 gpt2Model)
            {
                GgufExporter.Export(gpt2Model, tokenizer, path, name);
            }
            else
            {
                Console.WriteLine($"⚠ Экспорт не поддерживается для типа {llm.GetType().Name}");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка экспорта: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // ИНФОРМАЦИЯ О МОДЕЛИ
    // ═══════════════════════════════════════════════════════════
    static void ShowModelInfo(ILLM llm)
    {
        Console.WriteLine( "╔══════════════════════════════════════╗");
        Console.WriteLine( "║       ИНФОРМАЦИЯ О МОДЕЛИ            ║");
        Console.WriteLine( "╠══════════════════════════════════════╣");
        Console.WriteLine($"║ Архитектура: {llm.NetworkDescription()}");
        Console.WriteLine($"║ Vocab size:  {llm.Tokenizer.VocabSize}");
        Console.WriteLine($"║ Параметры:   {llm.TotalParameters():N0}");
        Console.WriteLine($"║ Emb dim:     {EmbeddingDim}");
        Console.WriteLine($"║ Hidden dim:  {HiddenDim}");
        Console.WriteLine($"║ Num heads:   {NumHeads}");
        Console.WriteLine($"║ Max seq len: {MaxSeqLen}");
        Console.WriteLine( "╚══════════════════════════════════════╝");
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

            var ids = tokenizer.Encode(input, addBos:false, addEos:false);
            string decoded = tokenizer.Decode(ids);

            Console.WriteLine($"  Token IDs [{ids.Count}]: [{string.Join(", ", ids)}]");

            // Показываем подслова
            var subwords = new List<string>();
            foreach (int id in ids)
            {
                string? tok = tokenizer.ToVocab().DecodeToken(id);
                subwords.Add(tok ?? "?");
            }
            Console.WriteLine($"  Подслова: [{string.Join("|", subwords)}]");
            Console.WriteLine($"  Decoded:  \"{decoded}\"");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // УТИЛИТЫ
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Конвертирует старый формат "User: ... Assistant: ... </s>"
    /// в формат с BPE спецтокенами (для pre-training данных).
    /// </summary>
    static string? FormatChatForBpe(string text)
    {
        // Пытаемся распарсить старый формат
        string clean = text.Replace("</s>", "").Trim();

        int userIdx = clean.IndexOf("User:", StringComparison.OrdinalIgnoreCase);
        int assistantIdx = clean.IndexOf("Assistant:",
            StringComparison.OrdinalIgnoreCase);

        if (userIdx >= 0 && assistantIdx > userIdx)
        {
            string userText = clean[(userIdx + 5)..assistantIdx].Trim()
                .TrimEnd(':').Trim();
            string assistantText = clean[(assistantIdx + 10)..].Trim()
                .TrimEnd(':').Trim();

            if (!string.IsNullOrEmpty(userText) &&
                !string.IsNullOrEmpty(assistantText))
            {
                // Базовый формат — конкретный токенизатор добавит свои маркеры
                //return $"<user> {userText} <assistant> {assistantText}";
                return $"<user> {userText}<assistant> {assistantText}";
            }
        }

        // Если не удалось распарсить, оборачиваем как есть
        if (!string.IsNullOrEmpty(clean))
            return clean.Trim();

        return null;
    }

    /// <summary>
    /// Токенизирует список текстов, фильтруя слишком короткие/длинные.
    /// </summary>
    static List<List<int>> TokenizeDataset(List<string> texts,
        ITokenizer tokenizer, int maxSeqLen)
    {
        return texts
            .Select(t => tokenizer.Encode(t, addBos:true, addEos:true))
            .Where(ids => ids.Count >= 2 && ids.Count < maxSeqLen)
            .ToList();
    }

    static string ResolvePath(string relativePath)
    {
        if (Path.IsPathRooted(relativePath))
            return relativePath;
        return Path.Combine(AppContext.BaseDirectory, relativePath);
    }




}