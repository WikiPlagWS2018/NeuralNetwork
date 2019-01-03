package org.apache.spark.examples.ml

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.sql.DataFrame

/*
Aktivierungsfunktion-Hidden: sigmoid
Aktivierungsfunktion-Out: softmax
MLPC employs backpropagation for learning the model.
Logistic loss function for optimization
L-BFGS as optimization routine.
*/
object NNTxt {

  val spark = SparkSession.builder.appName("MultilayerPerceptronClassifier").config("spark.master", "local").getOrCreate()

  def main(args: Array[String]): Unit = {

    val model = trainModel("/Users/davidketels/Documents/HTW/05_Semester/Wikiplag/Dtrain/sample_dtrain.txt", true)

    this.spark.stop()
  }

  def predict(model: MultilayerPerceptronClassificationModel, data: DataFrame): Unit = {
    val result = model.transform(data)
  }

  def trainModel(pathToDtrain: String, evaluate: Boolean): MultilayerPerceptronClassificationModel = {
    val dTestRatio = 0.5
    val dTrainRatio = 1 - dTestRatio
    val layers = Array[Int](2, 5, 5, 2) // input layer of size 2 (features) and output of size 2 (classes)
    val epochNum = 50
    val batchSize = 123

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = this.spark.read.format("libsvm").load(pathToDtrain)

    // Split the data into train and test
    val splits = data.randomSplit(Array(dTrainRatio, dTestRatio), seed = 1234L)
    val train = splits(0)
    //val test = splits(1)
    val test = splits(0)


    println(train.show())
    println(test.show())

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(batchSize)
      .setSeed(1234L)
      .setMaxIter(epochNum)

    // train the model
    val model = trainer.fit(train)

    if(evaluate) {
      // compute accuracy on the test set
      val result = model.transform(test)

      //val probability = result.drop("label").drop("features").drop("rawPrediction").drop("prediction")
      //println(probability.show())
      //println(probability.show(true))

      val predictionAndLabels = result.select("prediction", "label")

      val evaluator0 = new MulticlassClassificationEvaluator().setMetricName("accuracy")
      val evaluator1 = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
      val evaluator2 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
      val evaluator3 = new MulticlassClassificationEvaluator().setMetricName("f1")

      val accuracy = evaluator0.evaluate(predictionAndLabels)
      val precision = evaluator1.evaluate(predictionAndLabels)
      val recall = evaluator2.evaluate(predictionAndLabels)
      val f1 = evaluator3.evaluate(predictionAndLabels)

      println("Accuracy = " + accuracy)
      println("Precision = " + precision)
      println("Recall = " + recall)
      println("F1 = " + f1)
    }

    model
  }
}