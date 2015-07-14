package com.sdl.nplm.params

import org.canova.api.conf.Configuration
import org.deeplearning4j.nn.api.ParamInitializer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.Distributions
import org.deeplearning4j.nn.weights.WeightInitUtil
import org.nd4j.linalg.api.ndarray.INDArray
import com.sdl.nplm.conf.WordEmbeddingLayer
import org.deeplearning4j.nn.conf.MultiLayerConfiguration

/**
 * @author rorywaite
 */

object WordEmbeddingLayerParamInitializer extends ParamInitializer{
  val EMBEDDING_WEIGHTS = "nplmembedweights"
  
    override def init(params : java.util.Map[String, INDArray] , conf : NeuralNetConfiguration) = {
    val layer = conf.getLayer.asInstanceOf[WordEmbeddingLayer]
    val m = layer.embeddingDimension
    val dist = Distributions.createDistribution(conf.getDist)
    val C = WeightInitUtil.initWeights(Array(layer.vocabSize, m),conf.getWeightInit, dist);
    params.put(WordEmbeddingLayerParamInitializer.EMBEDDING_WEIGHTS, C)
  }
  
  override def init(params : java.util.Map[String, INDArray] , conf : NeuralNetConfiguration, extraConf : Configuration) 
    = init(params, conf)
}
