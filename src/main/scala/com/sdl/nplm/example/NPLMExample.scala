package com.sdl.nplm.example

import java.io.File
import java.io.FileWriter
import scala.io.Source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import com.sdl.spark.sql.sources.nplm.DefaultSource
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.`override`.ConfOverride
import com.sdl.nplm.conf.WordEmbeddingLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.lossfunctions.LossFunctions


/**
* A minimal example of the NPLM and a word embedding layer. 
* 
* The training data is not a typical language model task. We have
* - An input vocabulary of size 50000
* - An output vocabulary of size 2
* - The training data consists of 21-grams
* 
* The data is from a machine translation
* preordering task. More information can be found at
* 
* http://www.aclweb.org/anthology/N/N15/N15-1105.pdf
*  
*/
object NPLMExample extends App {
  val tmpFile = File.createTempFile("nplm_data", ".txt")
  tmpFile.deleteOnExit()
  val writer = new FileWriter(tmpFile)
  for (c <- Source.fromInputStream(this.getClass.getResourceAsStream("/nplm_data.txt"))) {
    writer.append(c)
  }
  writer.close

  val conf = new SparkConf()
    .setAppName("Preordering NPLM pipeline").setMaster("local[1]")
  val jsc = new SparkContext(conf);
  val jsql = new SQLContext(jsc);

  val path = "file://" + tmpFile.getAbsolutePath
  val data = jsql.read
    .format(classOf[DefaultSource].getName)
    .load(path);

  System.out.println("\nLoaded NPLM dataframe:")
  data.show(100);

  // prepare train/test set
  val trainingData = data.sample(false, 0.6, 11L)
  val testData = data.except(trainingData)

  val classification = new NeuralNetworkClassification()
    .setFeaturesCol("words")
    .setConf(getConf());
  val pipeline = new Pipeline().setStages(Array[PipelineStage](
    classification));

  // Fit the pipeline on training data.
  System.out.println("\nTraining...");
  val model = pipeline.fit(trainingData);

  // Make predictions on test data.
  System.out.println("\nTesting...");
  val predictions = model.transform(testData)

  System.out.println("\nTest Results:")
  predictions.show(100)

  def getConf(): MultiLayerConfiguration = {
    val seed = 6
    val iterations = 10

    new NeuralNetConfiguration.Builder()
      .layer(new RBM())
      .nIn(20)
      .nOut(2)
      .seed(seed)
      .iterations(iterations)
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0, 1e-1))
      .activationFunction("relu")
      .learningRate(1e-3)
      .l1(0.3)
      .constrainGradientToUnitNorm(true)
      .list(4)
      .backward(true)
      .pretrain(false)
      .hiddenLayerSizes(200, 50)
      .`override`(0, new ConfOverride() {
        def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder): Unit = {
          builder.layer(new WordEmbeddingLayer(10, 50000))
        }
      })
      .`override`(3, new ConfOverride() {
        def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder) = {
          builder.activationFunction("softmax");
          builder.layer(new OutputLayer());
          builder.lossFunction(LossFunctions.LossFunction.MCXENT);
        }
      })
      .build()
  }

} 