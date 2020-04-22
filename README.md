# Variational Autoencoder and Stochastic Variational Inference

## Intro
An implementation of the Variational Autoencoder based on [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) (Kingma and Welling 2013.).

VAE was trained and performed inference on binarized MNIST (handwritten digits) dataset.

## Model Definition
For an i.i.d. dataset with continuous latent variables per data point, the Variational Bayes algorithm optimizes a recognition model (encoder) that performs approximate posterior inference using ancestral sampling.

### Prior
Latent variable _z_ is sampled from prior distribution on _z_ usuing true parameters theta\*.

### Likelihood
Likelihood of data _x_ (i.e. all 784 pixels of image) is from a conditional distribution on z using true parameters theta\*. Here the distribution is a product of independent Bernoulli's whose means are outputted by the generator network (decoder) parameterized by theta.
