//https://github.com/virex-84

using System.Text.Json;

namespace LLM.ILGPU;

public enum DatasetType
{
    JSON,
    CSV
}

/// <summary>
/// Загрузчик датасетов для обучения
/// </summary>
public class Dataset
{
    public List<string> PretrainingData { get; }
    public List<string> ChatTrainingData { get; }

    public Dataset(string pretrainingDataPath, string chatTrainingDataPath, DatasetType typeOfData)
    {
        switch (typeOfData)
        {
            case DatasetType.CSV:
                PretrainingData = GetDataFromCsv(pretrainingDataPath);
                ChatTrainingData = GetDataFromCsv(chatTrainingDataPath);
                break;
            case DatasetType.JSON:
                PretrainingData = GetDataFromJson(pretrainingDataPath);
                ChatTrainingData = GetDataFromJson(chatTrainingDataPath);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(typeOfData), typeOfData, null);
        }
    }

    private static List<string> GetDataFromJson(string path)
    {
        var json = File.ReadAllText(path);
        var data = JsonSerializer.Deserialize<List<string>>(json);
        return data ?? new List<string>();
    }

    private static List<string> GetDataFromCsv(string path)
    {
        var data = new List<string>();
        var lines = File.ReadAllLines(path);
        
        foreach (var line in lines)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                data.Add(line.Trim());
            }
        }
        
        return data;
    }
}
