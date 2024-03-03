---
title: Convolutional Neural Networks
description: A Series of posts on the famous ConvNets.
tags:
  - neural
  - networks
  - convolution
  - image
  - recognition
  - pattern
  - recognition
  - deep
  - networks
hasplot: false
coll_name: conv_net
coll: neural_network
img: conv.png
---

## Introduction

On September 30, 2012, Alex Krizhevsky competed in the [ImageNet Large Scale Visual Recognition Challenge](http://image-net.org/challenges/LSVRC/2012/). His submission achieved a top-5 error rate of 15.3%. The runner up had an error rate of 26.1% giving a huge 10.8 points lead. What was so special about this model? He will go on to publish his model architechture and techniques in his paper, **AlexNet**. This would revolutionize the Deep Learning community that had been dormant for long. At the time of writing this(April 2020), AlexNet paper has been cited about 60,833 times. The one of the most important feature, among others, was that it was a Convolutional Neural Network.

This series will be devoted to learn about the Convolutional Neural Networks -  the mathematics, the implementations, the results. Nowadays, CNNs have become very popular in the visual learning and so it is mostly taught with image examples. I, however, would like to take other examples as well, so that the reader may not find it highly correlating with just the visual aspect of it and lose sight of the general notion of CNNs. 

We will first start with convolutions, more precisely, _discrete_ convolutions. We will try to understand what a convolution is. Why do we do convolutions at all? What does it mean to do a convolution? and the mathematics and other parts around it.

Check out [[convolutions]]