# Variational Autoencoder and Stochastic Variational Inference

## Intro
An implementation of the Variational Autoencoder based on [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) (Kingma and Welling 2013.).

VAE was trained and performed inference on binarized MNIST (handwritten digits) dataset.

## Model Definition
For an i.i.d. dataset with continuous latent variables per data point, the Variational Bayes algorithm optimizes a reconstruction network (encoder) that performs approximate posterior inference using ancestral sampling.

### Prior
Latent variable _z_ is sampled from prior distribution on _z_ usuing true parameters theta\*.

### Likelihood
Likelihood of data _x_ (i.e. all 784 pixels of image) is from a conditional distribution on z using true parameters theta\*. Here the distribution is a product of independent Bernoulli's whose means are outputted by the generator network (decoder) parameterized by theta.

## Variational Objective
Using an closed form expression for two Gaussians.

### KL divergence 
Like a regularization term, calculates the expected log ratio of approximate posterior from prior.

### Negative expectated reconstruction error
Negative log likelihood of datapoints.

## Visualizations
Post training, let's explore the properties of our trained approximate posterior.

#### Generative model
Sampling latent from prior then using the decoder (generative model) to compute the bernoulli means over the pixels of image given latent z:
![binary image sample](https://github.com/xwinxu/variational-mnist/blob/images/gen_samples.png)

#### Latent Posterior
Inferred latent posterior means from the encoder (reconstruction model):
![interpolate between latent rep of two points](https://github.com/xwinxu/variational-mnist/blob/images/latent_posterior.png)

#### Linear Interpolation
Generated samples from latent representations interpolated between the posterior means of two different training examples:
![sampled pairs 1-2, 3-8, 4-5](https://github.com/xwinxu/variational-mnist/blob/images/interpolated_means.png)

#### Joint distribution
Isocountours from joint distribution over latent z (note: 2D in this case, but can be higher for more expressivity) and trained top half of image x:
<img src="https://github.com/xwinxu/variational-mnist/blob/images/isocontours.png" alt="true and variational latent log posteriors" width="400" height="400" />

#### Creating Frankenstein images with optimized approximate posterior
Sample a latent z, feed into our probabilistic decoder, and infer the Bernouilli means of all bottom half of the image's pixels:
![Predicting the bottom half of an image from the top](https://github.com/xwinxu/variational-mnist/blob/images/frankenstein_bottom_to_top.png)
