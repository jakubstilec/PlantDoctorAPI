using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Reflection;

namespace PlantDoctorServer.ML
{
    public class ImageClassifier : IImageClassifier
    {
        readonly string ModelInput = "data";
        readonly string ModelOutput = "classLabel";
        const int ImageWidth = 224;
        const int ImageHeight = 224;
        const Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorsOrder ColorsOrder =
            Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorsOrder.ABGR;

        private Lazy<string> _modelFilePath;


        public ImageClassifier()
        {
            _modelFilePath = new Lazy<string>(() => SaveModelToTemp());
        }

        public string Classify(string imageFilePath)
        {
            MLContext mlContext = new MLContext();

            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image",
                    imageFolder: Path.GetDirectoryName(imageFilePath), inputColumnName: "ImagePath")
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image",
                    imageWidth: ImageWidth, imageHeight: ImageHeight, inputColumnName: "image"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "data",
                    inputColumnName: "image", orderOfExtraction: ColorsOrder))
                .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: _modelFilePath.Value,
                        outputColumnNames: new[] { ModelOutput },
                        inputColumnNames: new[] { ModelInput }));

            IEnumerable<ImageData> images = new[] { new ImageData() { ImagePath = Path.GetFileName(imageFilePath) } };
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

            var model = pipeline.Fit(imageData);

            IDataView scoredData = model.Transform(imageData);

            var tags = scoredData.GetColumn<string[]>("classLabel");
            var firstTag = tags.FirstOrDefault()?.FirstOrDefault();

            return firstTag;
        }

        private string SaveModelToTemp()
        {
            var modelFilePath = Path.GetTempFileName() + ".onnx";
            CopyEmbededResource("model.onnx", modelFilePath);
            return modelFilePath;
        }

        private async void CopyEmbededResource(string name, string outputFilePath)
        {
            if (File.Exists(outputFilePath))
                throw new InvalidOperationException("Model file already exists");

            var assembly = Assembly.GetExecutingAssembly();
            string resourcePath = assembly.GetManifestResourceNames().Single(str => str.EndsWith(name));

            using (Stream stream = assembly.GetManifestResourceStream(resourcePath))
            using (var fo = File.Create(outputFilePath))
            {
                await stream.CopyToAsync(fo);
            }
        }
    }
}
