//https://github.com/virex-84

using System.Collections.Concurrent;

namespace LLM.ILGPU;

/// <summary>
/// Словарь соответствий Токен <-> ID
/// </summary>
public class Vocab
{
    public ConcurrentDictionary<string, int> Encode { get; }
    public ConcurrentDictionary<int, string> Decode { get; }
    public List<string> Words { get; }

    public Vocab(List<string> words)
    {
        Words = words;
        Encode = new ConcurrentDictionary<string, int>();
        Decode = new ConcurrentDictionary<int, string>();
        for (int i = 0; i < words.Count; i++)
        {
            Encode[words[i]] = i;
            Decode[i] = words[i];
        }
    }

    public string? DecodeToken(int tokenId) =>
        Decode.TryGetValue(tokenId, out string? word) ? word : null;
}