package com.sdl.nplm.conf

import org.deeplearning4j.nn.conf.layers.Layer

/**
 * @author rorywaite
 */
case class NPLMLayer(order : Int, vocabSize : Int) extends Layer