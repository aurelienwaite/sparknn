package com.sdl.nplm.conf

import com.fasterxml.jackson.annotation.JsonTypeInfo
import org.deeplearning4j.nn.api.LayerFactory
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import com.sdl.nplm
import com.sdl.nplm.params.WordEmbeddingLayerParamInitializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.api.Layer

/**
 * @author rorywaite
 */
case class WordEmbeddingLayer(embeddingDimension: Int, vocabSize: Int) extends org.deeplearning4j.nn.conf.layers.Layer with LayerFactory[Layer] {

  def create[E <: org.deeplearning4j.nn.api.Layer](conf: NeuralNetConfiguration, listeners: java.util.Collection[IterationListener], index: Int): E =
    create[E](conf, index, 0, listeners)
  def create[E <: org.deeplearning4j.nn.api.Layer](conf: NeuralNetConfiguration): E =
    create[E](conf, 0, 0, new java.util.ArrayList[IterationListener])

  def create[E <: org.deeplearning4j.nn.api.Layer](conf: NeuralNetConfiguration, index: Int, numLayers: Int,
                                                   listeners: java.util.Collection[IterationListener]): E = {
    val ret = new com.sdl.nplm.WordEmbeddingLayer(conf)
    ret.setIterationListeners(listeners)
    ret.setIndex(index)
    val params = java.util.Collections.synchronizedMap(new java.util.LinkedHashMap[String, INDArray])
    initializer.init(params, conf)
    ret.setParamTable(params);
    ret.setConf(conf);
    ret.asInstanceOf[E]
  }

  def initializer(): org.deeplearning4j.nn.api.ParamInitializer = WordEmbeddingLayerParamInitializer

}