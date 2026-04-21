//https://github.com/virex-84

using System.Net;

namespace LLM.ILGPU;

/// <summary>
/// Общий интерфейс для всех языковых моделей в проекте.
/// Определяет базовые операции: предсказание, обучение, информация о модели.
/// </summary>
public interface ILLM : IDisposable
{
    // ═══ Свойства ═══

    /// <summary>Токенизатор, используемый моделью.</summary>
    ITokenizer Tokenizer { get; }

    /// <summary>Словарь для декодирования token ID → строка.</summary>
    Vocab Vocab { get; }

    /// <summary>Максимальная длина последовательности.</summary>
    int MaxSeqLen { get; }

    // ═══ Инференс ═══

    /// <summary>
    /// Генерирует ответ на входной текст.
    /// </summary>
    /// <param name="userInput">Текст пользователя.</param>
    /// <param name="temperature">Температура семплирования (0.0 = greedy, 1.0 = случайное).</param>
    /// <returns>Сгенерированный текст.</returns>
    string Predict(string userInput, float temperature = 0.7f);

    /// <summary>
    /// Генерирует ответ с дополнительным ограничением на длину.
    /// </summary>
    string PredictWithLimit(string userInput, float temperature = 0.7f,
        int? maxSeqLimit = null);

    // ═══ Обучение ═══

    /// <summary>
    /// Один шаг обучения на последовательности токенов.
    /// Возвращает значение loss.
    /// </summary>
    float TrainStep(List<int> tokens, float lr);

    // ═══ Информация ═══

    /// <summary>Общее количество параметров модели.</summary>
    int TotalParameters();

    /// <summary>Описание архитектуры (цепочка слоёв).</summary>
    string NetworkDescription();

    public void WarmUpInternal();

    List<ILayer> GetLayers();
}
