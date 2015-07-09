package com.sdl.nplm

import org.nd4j.api.linalg.DSL._
import org.deeplearning4j.nn.layers.BaseLayer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.gradient.Gradient
import org.nd4j.linalg.factory.Nd4j
import com.sdl.nplm.params.NPLMLayerParamInitializer
import org.deeplearning4j.optimize.api.IterationListener

import scala.collection.JavaConversions._

/**
 * Represents the input layer to a Neural Probabilistic Language Model (NPLM). See Bengio 2003 for details. 
 */
class NPLMLayer(conf: NeuralNetConfiguration) extends Layer{
 
  private var listeners = new java.util.ArrayList[IterationListener]()
  
  private var paramsTable : java.util.Map[String,INDArray] = null
  
  //The word embedding matrix
  private var C : INDArray = null;
  
  def activate(): INDArray = ???
  def activationMean(): INDArray = ???
  def backWard(x$1: org.deeplearning4j.nn.gradient.Gradient,x$2: org.deeplearning4j.nn.gradient.Gradient,x$3: INDArray,x$4: String): org.deeplearning4j.berkeley.Pair[org.deeplearning4j.nn.gradient.Gradient,org.deeplearning4j.nn.gradient.Gradient] = ???
  def backwardGradient(x$1: INDArray,x$2: org.deeplearning4j.nn.api.Layer,x$3: org.deeplearning4j.nn.gradient.Gradient,x$4: INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def calcGradient(x$1: org.deeplearning4j.nn.gradient.Gradient,x$2: org.nd4j.linalg.api.ndarray.INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def derivativeActivation(x$1: INDArray): INDArray = ???
  def error(x$1: INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def errorSignal(x$1: org.deeplearning4j.nn.gradient.Gradient,x$2: org.nd4j.linalg.api.ndarray.INDArray): org.deeplearning4j.nn.gradient.Gradient = ???
  def merge(x$1: org.deeplearning4j.nn.api.Layer,x$2: Int): Unit = ???
  def preOutput(x$1: INDArray): INDArray = ???
  def transpose(): org.deeplearning4j.nn.api.Layer = ???
  
  // Members declared in org.deeplearning4j.nn.api.Model
  def batchSize(): Int = ???
  def clear(): Unit = ???
  def conf(): NeuralNetConfiguration = ???
  def fit(x$1: INDArray): Unit = ???
  def getOptimizer(): org.deeplearning4j.optimize.api.ConvexOptimizer = ???
  def getParam(x$1: String): INDArray = ???
  def initParams(): Unit = ???
  def input(): INDArray = ???
  def paramTable(): java.util.Map[String,INDArray] = ???
  def setConf(x$1: NeuralNetConfiguration): Unit = ???
  def setParam(x$1: String,x$2: INDArray): Unit = ???
  def setParamTable(x$1: java.util.Map[String,INDArray]): Unit = ???
  def update(x$1: org.deeplearning4j.nn.gradient.Gradient): Unit = ???
  def validateInput(): Unit = ???
  
  
  def `type`() = Layer.Type.FEED_FORWARD
  
  def activate(input : INDArray) = {
    val nOut = conf.getNOut
    val confLayer = conf.getLayer.asInstanceOf[com.sdl.nplm.conf.NPLMLayer]
    val vocabSize = confLayer.vocabSize
    val order = confLayer.order
    val C = params
    val ret = Nd4j.create(Array[Int](vocabSize, order * nOut), Array[Float]())
    for (i <- 0 until confLayer.order) {
      val embedding = C ** input 
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
  
  def params : INDArray = Nd4j.toFlattened(C)
    
  def numParams() : Int = C.length
  
  def setParams(params : INDArray) =  C assign params
  
  def fit() = Unit
  
  def iterate(x$1: INDArray) = Unit
  
  def gradient(): org.deeplearning4j.nn.gradient.Gradient = null
  
  def gradientAndScore(): org.deeplearning4j.berkeley.Pair[org.deeplearning4j.nn.gradient.Gradient,java.lang.Double] = null

}