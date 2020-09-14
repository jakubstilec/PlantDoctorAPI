namespace PlantDoctorServer.ML
{
    public interface IImageClassifier
    {
        string Classify(string imageFilePath);
    }
}
