# LLMGPT2: Ультракомпактная реализация GPT-2 на C# с OpenCL и экспортом в GGUF

Реализация GPT-2 модели на C# с нуля и экспортом в формат GGUF.  Обучение с помощью OpenCL/CUDA/CPU через компонент ILGPU. Основная цель — создавать высокоэффективные модели, которые можно экспортировать в формат GGUF для инференса на `llama.cpp`, LM Studio и других современных средах для запуска LLM.

Финальная экспортированная модель GGUF весит всего **~440 КБ**, что делает её подходящей для сред с крайне ограниченными ресурсами.

<details>
  <summary>Нажмите, чтобы просмотреть изображение</summary>

  ![llama.cpp](llamacpp.png)

  ![LM Studio](lmstudio.png)
</details>

## Возможности

- **100% C#**: Определение модели, цикл обучения и экспортёр GGUF написаны на C# с использованием ILGPU.
- **Ускорение на OpenCL**: Обучение с ускорением на GPU через бэкенд OpenCL в ILGPU.
- **Экспорт в GGUF**: Нативный экспорт в формат GGUF v3 для максимальной совместимости.
- **Ультра-легковесная**: Финальная модель весит ~440 КБ, ~104 тыс. параметров.
- **Однооборотный чат без состояния**: Усечённый шаблон чата предотвращает накопление контекста, делая модель предсказуемым диалоговым агентом.

## Архитектура модели

Стандартный трансформер-декодер GPT-2 с минимальным количеством параметров.

### Параметры модели

| Параметр | Значение | Описание |
| :--- | :--- | :--- |
| `n_params` | ~104K | Общее количество обучаемых параметров |
| `n_vocab` | 512 | Размер словаря BPE |
| `n_ctx` | 63 | Максимальная длина контекста (адаптируется из данных) |
| `n_embd` | 64 | Размерность эмбеддингов |
| `n_head` | 2 | Количество голов внимания |
| `n_layer` | 1 | Количество блоков трансформера |
| `head_dim` | 32 | Размерность на одну голову (`n_embd / n_head`) |
| `ffn_dim` | 128 | Скрытая размерность в Feed-Forward слое |

**Разбивка параметров:**
- Эмбеддинги токенов: 32,768  (`512 × 64`)
- Позиционные эмбеддинги: 4,032  (`63 × 64`)
- Блок трансформера (1×): 33,472
  - Внимание (Q/K/V + Wo): 16,640
  - Два LayerNorm: 256
  - FFN (GELU, два линейных слоя): 16,576
- Финальный LayerNorm: 128
- Проекция головы LM: 33,280  (`64 × 512 + 512`)
- **Итого: 103,680 параметров**

### Структура слоёв (последовательная)

```
Входные ID
  ↓
Токеновые эмбеддинги (wte)           [vocab=512, dim=64]
  ↓
+ Позиционные эмбеддинги (wpe)    [seq_len=63, dim=64]  ← обучаемые + синусоидальные
  ↓
×1 × TransformerBlockPreNorm
  ├─ LayerNorm 1 (attn_norm)    [dim=64]
  ├─ MultiHeadAttention         [qkv объединены, 2 головы, каузальная маска]
  ├─ Остаточное сложение
  ├─ LayerNorm 2 (ffn_norm)     [dim=64]
   ├─ FeedForward (GELU)         [64 → 128 → 64]
  └─ Остаточное сложение
  ↓
Финальный LayerNorm (ln_f)          [dim=64]
  ↓
Линейная голова LM (lm_head)        [64 → 512]
  ↓
Логиты → Softmax → Токен
```

### Композиция блока трансформера (Pre-Norm)

Модель содержит **один** блок трансформера, следующий архитектуре **Pre-LayerNorm**:

```
x ──┐
    │
    ├─→ LayerNorm(x) ─→ MultiHeadAttention(Q,K,V) ──┐
    │                                              │
    └──────────────────────+───────────────────────┘
                           │
                           v
                     x + attention_output
                           │
                           ├─→ LayerNorm ─→ GELU(Linear_up) ─→ Linear_down
                           │
                           └──────────────────────+───────────────┘
                                                  │
                                                  v
                                            x + ffn_output
```

**Детали внимания:**
- **Каузальное селф-внимание** с маской для предотвращения заглядывания вперёд (треугольная)
- Проекции Q/K/V: объединены в одно ядро `[seq_len, 3*emb_dim]`
- Размерность головы: `emb_dim / num_heads = 32`
- Коэффициент масштабирования: `1/√32 ≈ 0.1768`
- Выходная проекция: отдельный линейный слой

**Детали Feed-Forward:**
- Двухслойный MLP: `Linear(emb→hidden) → GELU → Linear(hidden→emb)`
- Скрытая размерность: 128 (2× эмбеддинга)
- Смещения включены во всех линейных слоях
- Активация GELU: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))`

## Данные для обучения

Двухэтапный пайплайн обучения на пользовательских русскоязычных датасетах.

### Корпус для предварительного обучения

- **Размер:** ~2 КБ (24 предложения)
- **Содержание:** Базовые фактические утверждения о природных явлениях
- **Цель:** Обучить фундаментальной структуре языка и взаимосвязям слов
- **Пример:** "солнце восходит на востоке и садится на западе"

### Корпус для дообучения (диалоги)

- **Размер:** ~8 КБ (54 однооборотных пар вопрос-ответ)
- **Содержание:** Приветствия, представления и простые диалоговые обмены
- **Цель:** Адаптировать модель для однооборотного диалога в заданном формате

**Общий объём данных для обучения:** ~10 КБ необработанного текста.

## Шаблон чата и поведение без состояния

Определяющей особенностью является намеренно **бесстатусный** шаблон чата. В отличие от типичных чат-моделей, которые накапливают историю разговора, эта модель видит только **последнее** сообщение пользователя.

**Шаблон, хранящийся в метаданных GGUF:**
```
{% if messages %}
{% set message = messages[-1] %}
{% if message['role'] == 'user' %}
<user> {{ message['content'] }}<assistant>
{% endif %}
{% endif %}
```

**Формат во время выполнения:**
```
<user> {user_input}<assistant>
```

Прошлые обороты диалога не включаются в окно контекста. Это обеспечивает:
- Отсутствие памяти о предыдущих взаимодействиях
- Чистое следование однооборотным инструкциям
- Предсказуемое поведение независимо от истории чата
- Предотвращение "загрязнения" контекста в длительных беседах

Используемые специальные токены:
| Токен | ID | Роль |
| :--- | :--- | :--- |
| `<pad>` | 0 | Заполнитель |
| `<unk>` | 1 | Неизвестный |
| `<s>` | 2 | BOS (добавляется токенизатором) |
| `</s>` | 3 | EOS (добавляется в обучающих данных) |
| `<user>` | 4 | Маркер хода пользователя |
| `<assistant>` | 5 | Начало ответа ассистента |
| `<sep>` | 6 | Разделитель (не используется) |
| `<sep>` | 6 | Разделитель (не используется) |

## Токенизация

- **Алгоритм:** Byte-Pair Encoding (BPE)
- **Предварительная токенизация:** GPT-2 regex с учётом специальных токенов
- **Отображение байтов:** GPT-2 `bytes_to_unicode()` для полного покрытия UTF-8
- **Начальный словарь:** Специальные токены (7) + 256 байтовых токенов = 263 базовых токена
- **Слияния:** Изучаются инкрементально из обучающего корпуса
- **Конечный размер словаря:** 512 токенов (пользовательский BPE-словарь для диалогов)

Тренер BPE принудительно делает специальные токены атомарными единицами, предотвращая их разделение во время кодирования.

## Экспорт в GGUF

`GgufExporter` создаёт полностью совместимый файл **GGUF v3**:

- **Архитектура:** `gpt2`
- **Длина контекста:** 63 токена (адаптировано из 95-го перцентиля корпуса + округление)
- **Формат тензоров:** FP32 (все тензоры хранятся как 32-битные числа с плавающей запятой)
- **Словарь:** Стандартный байтовый BPE GPT-2 с пользовательскими специальными токенами
- **Метаданные:** Включают шаблон чата, ID токенов, слияния и оценки
- **Переназначение тензоров:** Словарь переупорядочен для соответствия стандартной раскладке GPT-2 (специальные токены 0–6, затем байтовые токены в каноническом порядке, затем слияния)
- **Позиционные эмбеддинги:** Комбинированные синусоидальные + обучаемые (суммируются, затем сохраняются как `position_embd.weight`)
- **Без квантизации:** Полная точность (модель и так крошечная)

**Экспортируемые тензоры:**
```
token_embd.weight          [64, 512]
position_embd.weight       [64, 63]

blk.0.attn_norm.weight     [64]
blk.0.attn_norm.bias       [64]
blk.0.attn_qkv.weight      [64, 192]   (Q/K/V объединены, транспонированы)
blk.0.attn_qkv.bias        [192]
blk.0.attn_output.weight   [64, 64]    (Wo)
blk.0.attn_output.bias     [64]
blk.0.ffn_norm.weight      [64]
blk.0.ffn_norm.bias        [64]
blk.0.ffn_up.weight        [64, 128]   (W1)
blk.0.ffn_up.bias          [128]
blk.0.ffn_down.weight      [128, 64]   (W2)
blk.0.ffn_down.bias        [64]

output_norm.weight         [64]
output_norm.bias           [64]
output.weight              [64, 512]   (lm_head, транспонирована)
output.bias                [512]
```

**Размер конечного файла:** ~440 КБ (103,680 параметров × 4 байта × overhead ~6%).

## Технический стек

| Компонент | Технология |
| :--- | :--- |
| Язык | C# 12 / .NET 8 |
| GPU бэкенд | ILGPU 1.5.3 (OpenCL) |
| Вычисления | ILGPU.Algorithms |
| Токенизатор | Пользовательский BPE (совместимый с GPT-2) |
| Формат экспорта | GGUF v3 |
| Целевые среды выполнения | llama.cpp, LM Studio, Ollama (via conversion) |

## Структура проекта

```
LLMGPT2/
├── src/
│   ├── GPT2/
│   │   ├── LLMGPT2.cs           # Основной класс модели (обучение + инференс)
│   │   └── GgufExporter.cs      # Экспортёр в GGUF v3
│   ├── Layers/
│   │   ├── EmbeddingLayer.cs
│   │   ├── PositionalEmbeddingLayer.cs  (обучаемые + синусоидальные)
│   │   ├── TransformerBlockPreNorm.cs   # Блок с Pre-Norm
│   │   ├── MultiHeadAttentionLayer.cs   # Каузальное внимание
│   │   ├── LayerNormLayer.cs
│   │   ├── GELUFeedForwardLayer.cs      # MLP с GELU
│   │   ├── LinearLayer.cs
│   │   └── LayerWeights.cs
│   ├── Tokenizers/
│   │   ├── BPETokenizer.cs      # BPE в стиле GPT-2
│   │   └── ITokenizer.cs
│   ├── ModelProfile.cs          # Автоконфигурация + анализатор корпуса
│   ├── Vocab.cs                 # Отображение токен ↔ ID
│   ├── Context.cs               # Обёртка для ILGPU context
│   ├── DatasetLoader.cs         # Загрузчик JSON/CSV
│   ├── Adam.cs                  # Оптимизатор Adam
│   ├── LossManager.cs           # Кросс-энтропия + softmax
│   └── MatrixOps.cs             # Ядра GPU (softmax, clip, etc.)
├── Program.cs                   # Интерактивный CLI
├── data/
│   ├── pretraining_data.json   # 24 предложения
│   └── chat_training_data.json # 54 Q&A pairs
├── LLMGPT2.csproj              # Пакеты ILGPU
└── README.md                   # Этот файл
```

## Быстрый старт

### Требования

- **.NET 8 SDK** или новее
- **GPU, совместимый с OpenCL 1.2+** (AMD, Intel, NVIDIA)
- Windows 10/11, Linux, or macOS

### Сборка и запуск

```bash
# Клонируйте и войдите в проект
cd LLMGPT2

# Восстановите пакеты
dotnet restore

# Сборка
dotnet build --configuration Release

# Запустите интерактивный CLI
dotnet run --project LLMGPT2.csproj
```

### Процесс обучения

1. **Создать новую модель** → токенизатор BPE обучается на вашем датасете
2. **Выбрать фазы обучения** → выберите предварительное обучение, дообучение, или оба
3. **Обучение** → модель учится на корпусе (гиперпараметры авто-сконфигурированы)
4. **Экспорт в GGUF** → `model.gguf` готов для `llama.cpp`

## Инференс с GGUF

### Использование `llama.cpp` (CLI)

```bash
llama-cli -m model.gguf -c 2048
```

### Использование LM Studio

1. Перетащите `model.gguf` в папку моделей LM Studio
2. Выберите модель
3. Начните чат — он будет однооборотным только (без памяти)


## Как это работает

### Прямой проход (GPU)

1. **ID токенов** → `EmbeddingLayer` (lookup) → `[seq_len, emb_dim]`
2. Добавить **синусоидальные + обучаемые** позиционные кодировки
3. Для каждого Transformer block:
   - **Attn:** Q,K,V = Linear(x) → reshape heads → causal attention (Q·K/√d → softmax → attn·V) → Wo
   - **Add residual** (x + attn_out)
   - **FFN:** LayerNorm → Linear → GELU → Linear
   - **Add residual** (x + ffn_out)
4. **Финальный LayerNorm**
5. **Linear** → logits over vocabulary

### Обратный проход (GPU)

1. Compute cross-entropy loss (neg log-likelihood)
2. Backprop through LM head (softmax + linear gradient)
3. Reverse through each block:
   - Accumulate gradients from residual branch
   - Backward through FFN (W2 → GELU → W1)
   - Backward through attention (Wo → softmax → QKV)
4. Update weights via **Adam optimizer**
5. Gradient clipping (max norm = 2.0–5.0 depending on dataset size)

### Процесс экспорта в GGUF

1. **Infer config** from layer list (count blocks, read dimensions)
2. **Rebuild GPT-2 standard vocab** with remapping:
   - IDs 0–6: special tokens (`<pad>`, `<unk>`, `<s>`, `</s>`, `<user>`, `<assistant>`, `<sep>`)
   - IDs 7–262: 256 byte tokens in GPT-2 `bytes_to_unicode()` order
   - IDs 263+: BPE merges (preserving original merge rank as score)
3. **Collect all tensors** from GPU buffers:
   - Transpose weight matrices from `[in, out]` to `[out, in]` for GGML column-major layout
   - Reorder vocab rows in `token_embd.weight` and `output.weight` by remapping old→new IDs
   - Fuse sinusoidal into learnable positional embeddings
4. **Write file** in single pass:
   - Header (24 bytes)
   - KV metadata (including chat template)
   - Tensor info array
   - Padding to 32-byte alignment
   - Tensor data arrays (each aligned to 32 bytes)

No temporary file buffering — streamed directly to disk.

## Performance Notes

- **Memory:** ~3× parameters during training (weights + gradients + Adam moments)
  - 104K params × 4 bytes × 3 ≈ **1.2 MB** GPU memory (excluding buffers)
- **Training speed:** ~50–200 tokens/sec on integrated GPUs (depends on sequence length)
- **Inference speed:** ~15–50 tokens/sec on CPU (via llama.cpp)
- **Batch size:** 1 (no batching implemented — training is online)
- **Throughput:** Limited by data copying between CPU/GPU each step

## References

- [GPT-2 Original Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ILGPU Documentation](https://ilgpu.net/docs/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp/)

## License

MIT
