package com.sdl.nplm.example

import java.io.File
import java.io.FileWriter
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import scala.collection.JavaConversions.seqAsJavaList
import scala.io.Source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.SQLContext
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.`override`.ConfOverride
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassification
import org.nd4j.api.linalg.DSL.toRichNDArray
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import com.sdl.nplm.conf.WordEmbeddingLayer
import com.sdl.nplm.params.WordEmbeddingLayerParamInitializer
import com.sdl.spark.sql.sources.nplm.DefaultSource
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.spark.ml.classification.NeuralNetworkClassificationModel

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
object NPLMExample {
  def main(args: Array[String]) {

    //writeParamsAndExit()

    def getDataframe(fileName: String)(implicit jsql: SQLContext) = {
      val tmpFile = File.createTempFile("nplm_data", ".txt")
      tmpFile.deleteOnExit()
      val writer = new FileWriter(tmpFile)
      for (c <- Source.fromInputStream(this.getClass.getResourceAsStream(fileName))) {
        writer.append(c)
      }
      writer.close
      jsql.read
        .format(classOf[DefaultSource].getName)
        .load("file://" + tmpFile.getAbsolutePath)
    }

    val conf = new SparkConf()
      .setAppName("Preordering NPLM pipeline").setMaster("local[1]")
    val jsc = new SparkContext(conf)
    implicit val jsql = new SQLContext(jsc)

    // prepare train/test set
    val trainingData = getDataframe("/1minibatch.samples")
    System.out.println("\nLoaded NPLM dataframe:")
    trainingData.show(100);
    val testData = getDataframe("/2minibatch.samples")

    val classification = new NeuralNetworkClassification()
      .setFeaturesCol("words")
      .setConf(getConf());
    val pipeline = new Pipeline().setStages(Array[PipelineStage](
      classification))

    // Fit the pipeline on training data.
    System.out.println("\nTraining...");
    val model = pipeline.fit(trainingData)

    val params = model.stages(0).asInstanceOf[NeuralNetworkClassificationModel].networkParams
    writeParamsAndExit(Option(params.value))

    // Make predictions on test data.
    System.out.println("\nTesting...");
    val predictions = model.transform(testData)
    predictions.printSchema()

    System.out.println("\nTest Results:")
    predictions.show(100)
    val logged = predictions.map { row =>
      val doLog: Int => Double =
        index => math.log10(row.getAs[org.apache.spark.mllib.linalg.Vector]("rawPrediction")(index))
      (doLog(0), doLog(1))
    }.collect
    for (l <- logged) println(l)
  }

  def getConf(): MultiLayerConfiguration = {
    val seed = 6
    val iterations = 1

    class MyBuilder(builder: NeuralNetConfiguration.Builder) {
      def configureLayer(): NeuralNetConfiguration.Builder =
        builder.iterations(iterations)
          .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
          .learningRate(1.0/64)
          .l1(0.0)
          .l2(0.0)
          .momentum(0.0)
          .constrainGradientToUnitNorm(false)
          .useAdaGrad(false)
    }
    
    implicit def toMyBuilder(builder: NeuralNetConfiguration.Builder) = new MyBuilder(builder)

    new NeuralNetConfiguration.Builder()
      .layer(new RBM())
      .nIn(20)
      .nOut(2)
      .seed(seed)
      .activationFunction("relu")
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0, 1e-1))
      .configureLayer()
      .list(4)
      .backward(true)
      .pretrain(false)
      //First layer is a dummy to be override by the word embedding layer
      .hiddenLayerSizes(1000, 100, 50)
      .`override`(0, new ConfOverride() {
        def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder): Unit = {
          builder.layer(new WordEmbeddingLayer(50, 30751))
            .activationFunction("identity").configureLayer()
        }
      })
      .`override`(3, new ConfOverride() {
        def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder) = {
          builder.layer(new OutputLayer()).activationFunction("softmax")
            .lossFunction(LossFunctions.LossFunction.MCXENT).configureLayer()
        }
      })
      .build()
  }

  /**
   * Used for initialising weights in nplm
   */
  def writeParamsAndExit(oParams: Option[INDArray] = None) = {
    val conf = getConf()
    val network = new MultiLayerNetwork(conf)
    network.init()
    network.setListeners(List(new ScoreIterationListener(1)))

    for (params <- oParams) network.setParams(params)

    implicit def weightsToStrings(params: INDArray): String =
      (for (i <- 0 until params.rows) yield (for (j <- 0 until params.columns)
        yield params(i, j).toString).mkString("\t")).mkString("\n") ++ "\n"

    val buffer = new StringBuilder

    Files.write(Paths.get("/tmp/params.txt"), network.params.getBytes(StandardCharsets.UTF_8))

    (buffer ++= "\n\\input_embeddings\n" ++= network.getLayer(0).getParam(WordEmbeddingLayerParamInitializer.EMBEDDING_WEIGHTS)
      ++= "\n\\hidden_weights 1\n" ++= network.getLayer(1).getParam(DefaultParamInitializer.WEIGHT_KEY).transpose()
      ++= "\n\\hidden_weights 2\n" ++= network.getLayer(2).getParam(DefaultParamInitializer.WEIGHT_KEY).transpose()
      ++= "\n\\output_weights\n" ++= network.getLayer(3).getParam(DefaultParamInitializer.WEIGHT_KEY).transpose()
      ++= "\n\\output_biases\n" ++= network.getLayer(3).getParam(DefaultParamInitializer.BIAS_KEY).transpose()
      ++= "\n\\end")
    Files.write(Paths.get("/tmp/nplm.txt"), buffer.toString.getBytes(StandardCharsets.UTF_8))

    System.exit(0)
  }

} 