import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

/*
Aktivierungsfunktion-Hidden-Layer: sigmoid
Aktivierungsfunktion-Out-Layer: softmax
backpropagation with Logistic loss function and L-BFGS as optimization routine.
*/

object NN {

  val spark = SparkSession.builder.appName("MultilayerPerceptronClassifier").config("spark.master", "local").getOrCreate()

  //Path to train Data
  val dataPath = "/Users/.../TrainData.txt" 

  //Meta Params
  val dValRatio = 1.toDouble / 3.toDouble
  val dTestRatio = 1.toDouble / 3.toDouble
  val dTrainRatio = 1.toDouble / 3.toDouble
  val layers = Array[Int](2,2,2,2) // input layer of size 2 (features) and output of size 2 (classes)
  val epochNum = 500
  val batchSize = 600 //--> SGD to GD

  //Evaluator
  val evaluator0 = new MulticlassClassificationEvaluator().setMetricName("accuracy")
  val evaluator1 = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision")
  val evaluator2 = new MulticlassClassificationEvaluator().setMetricName("weightedRecall")
  val evaluator3 = new MulticlassClassificationEvaluator().setMetricName("f1")

  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(batchSize).setSeed(3456L).setMaxIter(epochNum)

  //Only true if Meta Params are final
  val finalModel = true

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    if(!finalModel){
      trainAndValnModel()
    }
    else{
      val model = testModel()
      saveModel(model)
    }
    this.spark.stop()
  }

  /*
    fits Model with dTrain
    computed and print evaluation with dVal
   */
  def trainAndValnModel() = {
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = this.spark.read.format("libsvm").load(dataPath)

    // Split the data into train and test
    val splits = data.randomSplit(Array(dTrainRatio, dValRatio, dTestRatio), seed = 3456L)

    val trainData = splits(0)
    val valData = splits(1)
    val testData = splits(2)

    //println(trainData.show(), trainData.count())
    //println(valData.show(), valData.count())


    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(batchSize)
      .setSeed(3456L)
      .setMaxIter(epochNum)

    // train the model
    val model = trainer.fit(trainData)

    //Scores for val data
    val valResult = model.transform(valData)

    val predictionAndLabelsVal = valResult.select("prediction", "label")

    val accuracyVal = evaluator0.evaluate(predictionAndLabelsVal)
    val precisionVal = evaluator1.evaluate(predictionAndLabelsVal)
    val recallVal = evaluator2.evaluate(predictionAndLabelsVal)
    val f1Val = evaluator3.evaluate(predictionAndLabelsVal)

    println("Val Accuracy = " + accuracyVal)
    println("Val Precision = " + precisionVal)
    println("Val Recall = " + recallVal)
    println("Val F1 = " + f1Val+"/n")

  }

  /*
  fits Model with dTrain + dVal
  computed and print evaluation with dTest
 */
  def testModel(): MultilayerPerceptronClassificationModel = {
    val data = this.spark.read.format("libsvm").load(dataPath)

    // Split the data into train and test
    val splits = data.randomSplit(Array(dTrainRatio, dValRatio, dTestRatio), seed = 3456L)

    val trainData = splits(0).union(splits(1))
    val testData = splits(2)

    //train
    val model = trainer.fit(trainData)

    //test
    val testResult = model.transform(testData)
    val predictionAndLabels = testResult.select("prediction", "label")

    val accuracy = evaluator0.evaluate(predictionAndLabels)
    val precision = evaluator1.evaluate(predictionAndLabels)
    val recall = evaluator2.evaluate(predictionAndLabels)
    val f1 = evaluator3.evaluate(predictionAndLabels)

    println("Test Accuracy = " + accuracy)
    println("Test Precision = " + precision)
    println("Test Recall = " + recall)
    println("Test F1 = " + f1+"/n")
    model
  }

  def saveModel(model: MultilayerPerceptronClassificationModel) ={
    //TODO
  }

}
