package com.sdl.nplm.params

import org.canova.api.conf.Configuration
import org.deeplearning4j.nn.api.ParamInitializer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.Distributions
import org.deeplearning4j.nn.weights.WeightInitUtil
import org.nd4j.linalg.api.ndarray.INDArray

import com.sdl.nplm.conf.NPLMLayer

/**
 * @author rorywaite
 */

object NPLMLayerParamInitializer{
  val EMBEDDING_WEIGHTS = "nplmembedweights"
}

class NPLMLayerParamInitializer extends ParamInitializer{
  
  /**
   * This corresponds to the matrix C in Bengio,2003
   */ 
  
  
  override def init(params : java.util.Map[String, INDArray] , conf : NeuralNetConfiguration) = {
    val layer = conf.getLayer.asInstanceOf[NPLMLayer]
    val dist = Distributions.createDistribution(conf.getDist)
    val C = WeightInitUtil.initWeights(Array(conf.getNIn, layer.vocabSize),conf.getWeightInit, dist);
    params.put(NPLMLayerParamInitializer.EMBEDDING_WEIGHTS, C)
  }
  
  override def init(params : java.util.Map[String, INDArray] , conf : NeuralNetConfiguration, extraConf : Configuration) 
    = init(params, conf)
  
}