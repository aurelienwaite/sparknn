package com.sdl.nplm.conf

import org.deeplearning4j.nn.conf.layers.Layer

/**
 * @author rorywaite
 */
case class WordEmbeddingLayer(vocabSize : Int, embeddingDimension : Int) extends Layer