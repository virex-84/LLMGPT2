//https://github.com/virex-84

namespace LLM.ILGPU;

/// <summary>
/// Общий интерфейс для всех токенизаторов в проекте.
/// </summary>
public interface ITokenizer
{
    // ─── Properties ───
    int VocabSize { get; }
    int PadId { get; }
    int UnkId { get; }
    int BosId { get; }
    int EosId { get; }

    int UserId { get; }

    int AssistantId { get; }

    /// <summary>
    /// Словарь: токен → id. Используется при экспорте в GGUF.
    /// </summary>
    IReadOnlyDictionary<string, int> Encoder { get; }

    /// <summary>
    /// Список слияний (merge rules). Используется при экспорте в GGUF.
    /// </summary>
    IReadOnlyList<(string A, string B)> Merges { get; }

    // ─── Core Methods ───

    /// <summary>
    /// Кодирует текст в последовательность token IDs.
    /// Для обучения:  addBos = true, addEos = true
    /// Для инференса: addBos = true, addEos = false
    /// </summary>
    List<int> Encode(string text, bool addBos = false, bool addEos = false);

    /// <summary>
    /// Декодирует token IDs обратно в текст.
    /// </summary>
    string Decode(List<int> ids);

    /// <summary>
    /// Возвращает Vocab для совместимости с EmbeddingLayer.
    /// </summary>
    Vocab ToVocab();

    /// <summary>
    /// Сохраняет токенизатор в BinaryWriter (для сохранения модели).
    /// </summary>
    void SaveToStream(BinaryWriter writer);

    /// <summary>
    /// Загружает токенизатор из BinaryReader (для загрузки модели).
    /// </summary>
    static ITokenizer LoadFromStream(BinaryReader reader)
    {
        // По умолчанию используем BpeTokenizer.LoadFromStream
        return BPETokenizer.LoadFromStream(reader);
    }

    // ─── Static Helpers ───

    /// <summary>
    /// Форматирует текст для pretraining (без спецтокенов диалогов).
    /// Реализация зависит от класса токенизатора.
    /// </summary>
    string FormatPretraining(string text);

    /// <summary>
    /// Форматирует диалог в формат для токенизатора.
    /// Реализация зависит от типа токенизатора.
    /// </summary>
    string FormatDialogue(string user, string? assistant = null);
}
