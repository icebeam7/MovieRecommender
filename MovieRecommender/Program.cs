using System;
using System.IO;

using Microsoft.ML;
using Microsoft.ML.Trainers;

using MovieRecommender.Models;

namespace MovieRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            ITransformer model = BuildTrainModel(mlContext, trainingDataView);

            EvaluateModel(mlContext, testDataView, model);

            UseModelForSinglePrediction(mlContext, model);

            SaveModel(mlContext, trainingDataView.Schema, model);

            Console.ReadKey();
        }

        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var dataPath = Path.Combine(Environment.CurrentDirectory, "Data");
            var catalog = mlContext.Data;

            var trainingData = Path.Combine(dataPath, "recommendation-ratings-train.csv");
            var trainingDataView = catalog.LoadFromTextFile<MovieRating>(
                trainingData, hasHeader: true, separatorChar: ',');

            var testData = Path.Combine(dataPath, "recommendation-ratings-test.csv");
            var testDataView = catalog.LoadFromTextFile<MovieRating>(
                testData, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        public static ITransformer BuildTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            var catalog = mlContext.Transforms.Conversion;

            IEstimator<ITransformer> estimator = catalog.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(catalog.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

            var testInput = new MovieRating { userId = 6, movieId = 10 };

            var movieRatingPrediction = predictionEngine.Predict(testInput);

            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
                Console.WriteLine("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
            else
                Console.WriteLine("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var data = Path.Combine(Environment.CurrentDirectory, "Data");
            var modelPath = Path.Combine(data, "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }
    }
}
