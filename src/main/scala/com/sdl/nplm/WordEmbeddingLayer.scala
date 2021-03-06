package com.sdl.nplm

import org.nd4j.api.linalg.DSL._
import org.deeplearning4j.nn.layers.BaseLayer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.gradient.Gradient
import org.nd4j.linalg.factory.Nd4j
import com.sdl.nplm.params.WordEmbeddingLayerParamInitializer
import org.deeplearning4j.optimize.api.IterationListener
import scala.collection.JavaConversions._
import com.sdl.nplm.params.WordEmbeddingLayerParamInitializer
import com.sdl.nplm.params.WordEmbeddingLayerParamInitializer
import org.deeplearning4j.nn.params.PretrainParamInitializer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.gradient.DefaultGradient
import org.nd4j.linalg.indexing.NDArrayIndex
import org.deeplearning4j.optimize.Solver

/**
 * Represents the input layer to a Neural Probabilistic Language Model (NPLM).
 * This type of LM is a continuous space N-gram language model. The N-gram order
 * and vocabulary size |V| are required configuration parameters
 *
 * This parameters of this matrix are given by a |V| x m embedding matrix C.
 * Each of the N-1 words in the N-gram history are represented as N-1 one-hot input
 * vectors of size |V|. The matrix C transforms each vector to an m-dimensional
 * vector which are concatenated to create a hidden layer of size m|v|. No activation
 * function is used for the hidden layer.
 *
 * This layer is designed to be used as the input to a multi-layer feed forward network.
 *
 * See Bengio 2003 for details.
 */
class WordEmbeddingLayer(var conf: NeuralNetConfiguration) extends Layer {

  val `type` = Layer.Type.FEED_FORWARD

  private var listeners = new java.util.ArrayList[IterationListener]()

  var paramTable: java.util.Map[String, INDArray] = null

  private var index = 0

  lazy val solver = new Solver.Builder()
    .model(this).configure(conf)
    .build().getOptimizer()

  /**
   * The input is a vector of word indices. It is a compact representation of the very sparse
   * one hot word vectors that form the input. We need to keep track of the input after activation
   * because it is used to determine which columns of the embedding matrix are updated.
   */
  var input: INDArray = null

  def activate(): INDArray = ???
  def activationMean(): INDArray = ???
  def backWard(x$1: org.deeplearning4j.nn.gradient.Gradient, x$2: org.deeplearning4j.nn.gradient.Gradient, x$3: INDArray, x$4: String): org.deeplearning4j.berkeley.Pair[org.deeplearning4j.nn.gradient.Gradient, org.deeplearning4j.nn.gradient.Gradient] = ???
  def calcGradient(x$1: org.deeplearning4j.nn.gradient.Gradient, x$2: org.nd4j.linalg.api.ndarray.INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def derivativeActivation(x$1: INDArray): INDArray = ???
  def error(x$1: INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def errorSignal(x$1: org.deeplearning4j.nn.gradient.Gradient, x$2: org.nd4j.linalg.api.ndarray.INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def merge(x$1: org.deeplearning4j.nn.api.Layer, x$2: Int): Unit = ???
  def transpose(): org.deeplearning4j.nn.api.Layer = ???

  // Members declared in org.deeplearning4j.nn.api.Model
  def update(x$1: org.deeplearning4j.nn.gradient.Gradient): Unit = ???

  def initParams(): Unit = WordEmbeddingLayerParamInitializer.init(paramTable, conf)

  def getParam(key: String): INDArray = paramTable(key)

  /**
   * Perform step 1a) from the algorithm in Bengio 2003 p 1145
   */
  def activate(input: INDArray) = {
    this.input = input
    val nIn = conf.getNOut
    val confLayer = conf.getLayer.asInstanceOf[com.sdl.nplm.conf.WordEmbeddingLayer]
    val vocabSize = confLayer.vocabSize
    val m = confLayer.embeddingDimension
    val C = paramTable(WordEmbeddingLayerParamInitializer.EMBEDDING_WEIGHTS)
    val ret = Nd4j.create(input.rows, m * input.columns)
    for (i <- 0 until input.rows; j <- 0 until input.columns) {
      // Concatenate word embedding vectors
      val wordIndex = input(i, j).toInt
      for (k <- 0 until m)
        ret.put(i, m * j + k, C.getRow(wordIndex)(k))
    }
    ret
  }

  def getIterationListeners(): java.util.Collection[IterationListener] = listeners

  def setIterationListeners(listeners: java.util.Collection[IterationListener]) =
    this.listeners = new java.util.ArrayList(listeners)

  def accumulateScore(x$1: Double) = Unit

  def score() = 0

  def setScore() = Unit

  def transform(data: INDArray): INDArray = activate(data)

  def params: INDArray = Nd4j.toFlattened(paramTable.values)

  def numParams(): Int = paramTable.values.map(_.length).reduce(_ + _)

  /**
   * Perform step 2d) from the algorithm in Bengio 2003 p 1146
   */
  def update(gradient: INDArray, paramType: String): Unit = {
    val C = paramTable.get(WordEmbeddingLayerParamInitializer.EMBEDDING_WEIGHTS)
    C -= gradient
  }

  /**
   * Aggregate the derivatives from the hidden layers
   */
  def backwardGradient(z: INDArray, nextLayer: Layer, gradient: Gradient, activation: INDArray): Gradient = {
    val dW = gradient.gradientForVariable.getOrElse(DefaultParamInitializer.WEIGHT_KEY,
      sys.error("Can't find gradient for " + DefaultParamInitializer.WEIGHT_KEY))
    val cl = conf.getLayer.asInstanceOf[com.sdl.nplm.conf.WordEmbeddingLayer]
    val m = cl.embeddingDimension
    val dC = Nd4j.zeros(cl.vocabSize, m)
    val ret = new DefaultGradient
    ret.setGradientFor(WordEmbeddingLayerParamInitializer.EMBEDDING_WEIGHTS, dC)
    for (i <- 0 until activation.rows; j <- 0 until activation.columns) {
      val wordIndex = activation(i, j).toInt
      val embedding = dW.getRow(i).get(NDArrayIndex.interval(j * m, j * m + m))
      dC.getRow(wordIndex) += embedding
    }
    ret
  }

  def setParams(params: INDArray) = {
    val C = paramTable.get(WordEmbeddingLayerParamInitializer.EMBEDDING_WEIGHTS)
    if (params.length() != C.length)
      sys.error("Unable to set parameters: must be of length " + C.length)
    val cl = conf.getLayer.asInstanceOf[com.sdl.nplm.conf.WordEmbeddingLayer]
    val m = cl.embeddingDimension
    val vocabSize = cl.vocabSize
    C.assign(params.reshape(vocabSize, m))
  }

  def fit() = Unit

  def fit(x$1: INDArray): Unit = Unit

  def iterate(x$1: INDArray) = Unit

  def gradient(): org.deeplearning4j.nn.gradient.Gradient = null

  def gradientAndScore(): org.deeplearning4j.berkeley.Pair[org.deeplearning4j.nn.gradient.Gradient, java.lang.Double] = null

  def getIndex(): Int = index

  def setIndex(index: Int): Unit = this.index = index

  def setParam(key: String, value: INDArray): Unit = paramTable(key) = value

  def setParamTable(pt: java.util.Map[String, INDArray]): Unit = paramTable = pt

  def validateInput(): Unit = Unit

  def batchSize(): Int = 0

  def setConf(conf: NeuralNetConfiguration): Unit = this.conf = conf

  def clear(): Unit = Unit

  def getOptimizer(): org.deeplearning4j.optimize.api.ConvexOptimizer = solver

  def preOutput(input: INDArray): INDArray = activate(input)

}