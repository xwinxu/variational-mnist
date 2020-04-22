# Implements auto-encoding variational Bayes.

from __future__ import absolute_import, division
from __future__ import print_function
import jax.numpy as np
import jax.random as random
import numpy as onp
from jax.scipy.stats import norm
from jax.nn import sigmoid
from jax import vmap, grad, value_and_grad, jit, tree_util
from jax.experimental.optimizers import adam

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import pickle
from data import load_mnist, save_images


def diag_gaussian_log_density(x, mu, log_sigma):
  """
  Args:
    x: random variable
    mu: mean
    log_sigma: log standard deviation
  Return:
    log normal density.
  """
  assert x.ndim == 1
  return np.sum(norm.logpdf(x, mu, np.exp(log_sigma)), axis=-1)


def unpack_gaussian_params(params):
  """
  Args: 
    params of a diagonal Gaussian.
  Return:
    mean, log standard deviation
  """
  D = np.shape(params)[-1] // 2
  print("params shape", params.shape)
  mu, log_sigma = params[:D], params[D:]
  return mu, log_sigma


def sample_diag_gaussian(mu, log_std, subkey):
  """Reparameterization trick for getting z from x.
  """
  return random.normal(subkey, mu.shape) * np.exp(log_std) + mu


def bernoulli_log_density(b, unnormalized_logprob):
  """
  Args: 
    Unnormalized_logprob: log(mu / (1 - mu)) <- "logit"
    b: 0 or 1 (i.e. binarized digit/image)
  Return: 
    log Ber(b | mu)
  """
  s = b * 2 - 1
  return -np.logaddexp(0., -s * unnormalized_logprob)


def init_net_params(scale, layer_sizes, key):
  """
  Args:
    scale: scaling factor
    layer_sizes: List[number of neurons per layer]
  Return:
    Tuple[weights, biases] for all layers."""
  k1, k2 = random.split(key, 2)
  return [
      (
          scale * random.normal(k1, (m, n)),  # weight matrix
          scale * random.normal(k2, (n,)))  # bias vector
      for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
  ]


def neural_net_predict(params, inputs):
  """
  Args:
    params: List[Tuple(weights, bias)]
    inputs: an (N x D) matrix, (batch 2D latent vector Dz x B), here Dz = 2
  Return:
    Applies batch normalization to every layer but the last."""
  for W, b in params:
    print(f"W {W.shape} b {b.shape}")
    if W.shape[0] == inputs.shape[0]:
      outputs = np.dot(inputs.T, W) + b
    else:
      outputs = np.dot(inputs, W) + b  # linear transformation
    inputs = np.tanh(outputs)  # nonlinear transformation
  return outputs


def nn_predict_gaussian(params, inputs):
  """
  Args:
    params: variational parameters
    inputs: batch of images
  Return:
    means and diagonal variances
  """
  return unpack_gaussian_params(neural_net_predict(params, inputs))


def log_prior(z):
  """
  Args:
    z: latent variable
  Return:
    computes log of pior over digit's latent represenation z.
  """
  assert z.ndim == 1
  return diag_gaussian_log_density(z, 0, 0)


def decoder(z, params):
  """
  Args:
    z: latent representation
    params: decoder network parameters, theta
  Return: 
    784-D (28x28 pixels) mean vector of prod of Bern
  """
  logits = neural_net_predict(params, z)
  return logits


def log_likelihood(z, x, params):
  """
  Args:
    z: latent representation
    x: binarized digit
    params: logits from decoder/generator theta
  Return: 
    log likelihood log p(x|z), p image given latent
  """
  mu = decoder(z, params)  # logits
  likelihood = bernoulli_log_density(x, mu)
  assert likelihood.ndim == 1
  return np.sum(likelihood) # sum over pixels


def generate_from_prior(gen_params,
                        num_samples,
                        noise_dim,
                        key=random.PRNGKey(2)):
  """
  Args:
    gen_params: decoder parameters
    num_samples: number of latent variable samples
  Return: 
    Fake data: Bernouilli means p(x|z)
  """
  latents = random.normal(key, (num_samples, noise_dim))
  return sigmoid(neural_net_predict(gen_params, latents))


def joint_log_density(x, z, params):
  """
  Args:
      z: latent representation
      x: binarized digit
    params: logits from decoder
  Return: 
    log p(z, x) for a single image
  """
  return log_prior(z) + log_likelihood(z, x, params)


def encoder(x, params):
  """
  Args:
    x: batch of images
    params: variational parameters mu and sigma
    phi: recognition parameters
  Return: 
    mean and log std of factorized Gaussian with D = 2
  """
  mu, log_sigma = nn_predict_gaussian(params, x)
  return mu, log_sigma


def log_q(z, mu, log_sigma):
  """
  Args:
    z: latent representation
    mu, log_sigma: variational distribution parameters
  Return: 
    p(x|params) likelihood of x
  """
  return diag_gaussian_log_density(z, mu, log_sigma)


def elbo(x, params, subkey):
  """
  Args:
    x: batch of B images, D_x x B
      only need to sample single z for each image in the batch
    params: {encoder (recognition network): encoder_params phi,
             decoder (likelihood): decoder_params theta}
    subkey: jax random key
  Return: 
    scalar, unbiased estimate of mean variaitonal elbo on images
  """
  encoder_params, decoder_params = params['enc'], params['dec']
  # latent means and log stds
  mu_qz, log_sigma_qz = encoder(x, encoder_params) 
  # Monte Carlo est of KL divergence of q from prior p (both Gaussian)
  # KL(q(z | x) || p(z)),  q ~ N(z | mu(x), sigma(x)) and p ~ N(0, I_DzxDz)
  kl = -1 / 2 * np.sum(
      np.log(np.square(np.exp(log_sigma_qz))) + 1 -
      np.square(np.exp(log_sigma_qz)) - np.square(mu_qz))
  # latent variables
  z = sample_diag_gaussian(mu_qz, log_sigma_qz, subkey)
  # p(image x | latents z)
  ll = log_likelihood(z, x, decoder_params)

  return ll - kl


def loss(*args, **kwargs):
  # Note: negate ll for the elbo loss to minimize
  return -elbo(*args, **kwargs)


def batch_loss(*args, **kwargs):
  """Negative elbo estimate over batch of data."""
  loss_ = vmap(
      loss, in_axes=(0, None, 0))(*args,
                                  **kwargs)  # correspond each sample with input
  return np.mean(loss_)


############## TRAINING VAE ##############

def load_data():
  """Binarized training data; first 10k for train, second 10k for testing."""
  N, train_images, train_labels, _, _ = load_mnist()
  print("Loading training data...")

  print(
      f"MNIST loaded train: {train_images.shape} labels: {train_labels.shape}")

  def binarise(images):
    on = images > 0.5
    images = images * 0.0
    images[on] = 1.0
    return images

  print("Binarising training data...")
  train_images = binarise(train_images)
  train_images_, train_labels_ = train_images[0:10000], train_labels[0:10000]
  test_images_, test_labels_ = train_images[10000:20000], train_labels[10000:
                                                                       20000]

  return train_images_, train_labels_, test_images_, test_labels_


def train(train_images, test_images, param_dump='opt-params.pkl', seed=0):
  """
  Optimize gradients of weights over batches of data with elbo estimate.
  """
  # Model hyper-parameters
  latent_dim = 2
  data_dim = 784  # How many pixels in each image (28x28).
  gen_layer_sizes = [latent_dim, 500, data_dim]  # decoder has 500 hidden
  rec_layer_sizes = [data_dim, 500, latent_dim * 2]  # encoder has 500 hidden

  # Training parameters
  param_scale = 0.01
  batch_size = 200
  num_epochs = 100  # train for 100 epochs
  learning_rate = 0.001

  key = random.PRNGKey(seed)
  key, enc_k, dec_k = random.split(key, 3)
  init_gen_params = init_net_params(param_scale, gen_layer_sizes,
                                    dec_k)  # encoder
  init_rec_params = init_net_params(param_scale, rec_layer_sizes,
                                    enc_k)  # decoder
  combined_init_params = dict(dec=init_gen_params, enc=init_rec_params)

  num_batches = int(np.ceil(len(train_images) / batch_size))

  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx + 1) * batch_size)

  objective_grad = jit(value_and_grad(batch_loss,
                                      argnums=1))  # differentiate w.r.t params

  opt_init, opt_update, opt_get_params = adam(step_size=learning_rate)
  opt_state = opt_init(combined_init_params)

  it = 0
  for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(range(num_batches)):
      batch_x = train_images[batch_indices(batch)]
      params = opt_get_params(opt_state)
      key, *subkeys = random.split(key, batch_size + 1)
      subkeys = np.stack(subkeys, axis=0)
      print("subkeys shape: ", subkeys.shape)
      loss_, grad_ = objective_grad(batch_x, params, subkeys)
      opt_state = opt_update(it, grad_, opt_state)

      if it % 100 == 0: # save samples during training
        gen_params, rec_params = params['dec'], params['enc']
        fake_data = generate_from_prior(gen_params, 20, latent_dim, key)
        save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)

      if it == 0 or (it + 1) % 100 == 0:
        test_size = test_images.shape[0]
        print("test size: ", test_images.shape, train_images.shape)
        key, *subkeys = random.split(key, test_size + 1)
        subkeys = np.stack(subkeys, axis=0)
        # print performance
        loss_t = batch_loss(test_images, params, subkeys)
        message = f"Epoch: {epoch} \t Batch: {batch} \t Loss: {loss_:.3f} \t Test Loss: {loss_t:.3f}"
        tqdm.write(message)
      it += 1

  # pickle to save trained weights
  params = opt_get_params(opt_state)
  with open(param_dump, 'wb') as file:
    pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

############## VISUALIZE APPROXIMATE POSTERIOR #############

def load_params(file='params2.pkl'):
  with open(file, 'rb') as f:
    params = pickle.load(f)
  # JAX does not recognize pickled file, must re-format
  # params: List[[Tuple(weights), Tuple(bias)]]
  num_layers = 2
  for k in ['dec', 'enc']:
    params[k] = list(params[k])
    for l in range(num_layers):
      params[k][l] = tuple(params[k][l])

  print("after loaded params", type(params), type(params['enc']),
        type(params['dec'][0]), type(params['dec'][0][0]))

  return params


def sample_gen(params, num_samples=10, seed=0):
  """
  Args: 
    params: the variational parameters
    num_samples: number of times to sample from distributino
    seed: random seed
  Plot samples from trained generative model using ancestral sampling.
  """
  key = random.PRNGKey(seed)
  key, k1, k2 = random.split(key, 3)
  # sample z from prior num_samples times
  # use generative model to compute bernouilli means over pixels of x given z
  means = generate_from_prior(params['dec'], num_samples, 2, k1)
  # plot means as greyscale image
  mean_images = means.reshape([-1, 28, 28])
  # sample binary image x from product of Bern and plot as image
  sample_means = random.bernoulli(k2, mean_images)
  # concatenate plots: row 1, bernouilli means, row 2 corresponding binary img sampled from 1
  plot_means = np.stack([mean_images, sample_means])
  image_ = onp.zeros([2 * 28, 10 * 28])
  num_rows = 2
  num_cols = 10
  for i in range(num_rows):
    for j in range(num_cols):
      image_[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = plot_means[i, j, ...]
  plt.imshow(image_, cmap=plt.cm.binary)
  plt.axis('off')
  plt.savefig('gen_samples.png', bbox_inches='tight')


def latent_means(params, train_images, train_labels):
  """
  Args:
    params: List[Tuple(W, b)] for each layer in NN
    train_images, train_labels = (10k, 784), (10k, 10)
  Latent space scatter plot, each point is a different image in training set.
  Visualizes which part of latent space corresponds to which kinds of data.
  """
  # encode each image in the train set
  mus, log_sigmas = vmap(
      encoder, in_axes=(0, None))(train_images, params['enc'])
  # one hot encode -> continuous
  labels = np.argmax(train_labels, axis=-1)
  num_labels = train_labels.shape[-1]

  # 2D mean vector of each encoding q_phi(z|x)
  # plot mean vectors in 2D latent space
  # color each point to class label (0, 9)
  def cmap_process(cmap, N):
    if type(cmap) == str:
      cmap = plt.get_cmap(cmap)
    col_idx = onp.concatenate((onp.linspace(0, 1., N), (0., 0., 0., 0.)))
    col_rgb = cmap(col_idx)
    idx = onp.linspace(0, 1., N + 1)
    cols = {}
    for k_i, k in enumerate(('red', 'green', 'blue')):
      cols[k] = [
          (idx[i], col_rgb[i - 1, k_i], col_rgb[i, k_i]) for i in range(N + 1)
      ]

    return mpl.colors.LinearSegmentedColormap(f"{cmap.name}-{N}", cols, 1024)

  def color_index(num_colors, cmap):
    cmap = cmap_process(cmap, num_colors)
    color_map = mpl.cm.ScalarMappable(cmap=cmap)
    color_map.set_array([])
    color_map.set_clim(-0.5, num_colors + 0.5)
    color_bar = plt.colorbar(color_map, fraction=0.045, pad=0.04)
    color_bar.set_ticks(onp.linspace(0, num_colors, num_colors))
    color_bar.set_ticklabels(range(num_colors))

    return color_bar

  fig, ax = plt.subplots()
  cmap = plt.cm.jet
  ax.scatter(mus[:, 0], mus[:, 1], c=labels, s=1, cmap=cmap)
  cb = color_index(num_labels, cmap)
  ratio = 1.0
  left, right = ax.get_xlim()
  low, hi = ax.get_ylim()
  ax.set_aspect(abs((right - left) / (low - hi)) * ratio)
  ax.set_xlabel(r'$\mu_z(x)_0$')
  ax.set_ylabel(r'$\mu_z(x)_1$')
  ax.set_title("Latent posterior mean given image")
  fig.set_size_inches([6, 6], forward=True)
  plt.savefig("latent_posterior.png", bbox_inches='tight')


def lin_interpolate(params, train_images, train_labels, examples):
  """
  Args:
    params: List[Tuple(W, b)] for each layer in NN
    train_images, train_labels = (10k, 784), (10k, 10)
    examples: List[Tuple[digit 1, digit 2]] samples to interpolate
  Examining latent variable model with continuous latent variables by 
  linearly interpolating between latent reps (mean vecs of encodings) of two points.
  """

  def interpolate(za, zb, alpha):
    """Linear interpolation z_alpha = alpha * z_a + (1-a) * z_b
    """
    z_alpha = alpha * za + (1 - alpha) * zb
    return z_alpha

  # sample 3 pairs of images, each having a different class
  labels_to_images = defaultdict(list)
  # encode data and get mean vectors
  labels = np.argmax(train_labels, axis=-1)
  # linearly interpolate between mean vectors
  for im, lab in tqdm(zip(train_images, labels)):
    labels_to_images[lab].append(im)
  print("labels to images", labels_to_images.keys())
  # plot Bernoulli means p(x|z_\alpha) at 10 equally spaced points
  image_ = onp.zeros([3 * 28, 10 * 28])
  # plot generative distribution along linear interpolation
  for row, pair in enumerate(examples):
    images = [labels_to_images[pair[0]][0], labels_to_images[pair[1]][0]]
    images = np.stack(images)
    mus, log_sigmas = vmap(encoder, in_axes=(0, None))(images, params['enc'])
    alphas = np.linspace(0, 1, 10)[::-1]
    interpolated_means = [interpolate(mus[0], mus[1], a) for a in alphas]
    interpolated_means = np.stack(interpolated_means)
    bern_mus = sigmoid(
        vmap(decoder, in_axes=(0, None))(interpolated_means, params['dec']))
    bern_ims = bern_mus.reshape([-1, 28, 28])
    print("bern ims", bern_ims.shape)
    for col in range(10):
      image_[row * 28:(row + 1) * 28, col * 28:(col + 1) *
             28] = bern_ims[col, ...]

  fig, ax = plt.subplots()
  plt.imshow(image_, cmap=plt.cm.binary)
  plt.axis('off')
  plt.savefig('interpolated_means.png', bbox_inches='tight')

############ STOCHASTIC VARIATIONAL INFERENCE #############

def top_half(x):
  """
  Args:
    x: image
  Return: 
    top half of 28x28 image array.
  """
  assert x.shape == (28, 28)
  return x[:14, :]


def log_like_top_half(x, z, params):
  """
  Args:
    z: latent vector
    x: image
    params: decoder parameters
  Return: 
    log p(top half of image x | z) integrated out exactly for
      all unobserved dimensions of x are leaf nodes since ll factorizes
  """
  x = x.reshape([28, 28])
  mu_logits = decoder(z, params)  # unnormalized_logprob
  mu_image = mu_logits.reshape([28, 28])
  image_top_half = top_half(x)
  mu_top_half = top_half(mu_image)
  bern_density = bernoulli_log_density(image_top_half, mu_top_half)
  return np.sum(bern_density)


def joint_ll_top_half(x, zs, params):
  """
  Args:
    x: image
    zs: array
    params; decoder parameters
  Return: 
    log joint density log p(z, top half image x) for each z
  """
  return log_prior(zs) + log_like_top_half(x, zs, params)


def init_var_params(subkey):
  """
  Args:
    subkey: jax key
  Return:
    Initialized variational parameters phi_mu and phi_logsigma for
    variational distribution q(z|top half of x).
  """
  return random.normal(subkey, (4,))


@jit
def elbo_half(*args, **kwargs):
  """
  ELBO estimate over K samples, batched for half of image x.
  """

  def elbo_k(x, qz_params, dec_params, subkey):
    """
    Estimate of ELBO over K samples z ~ q(z | top half of x).
    """
    mu_qz, log_sigma_qz = unpack_gaussian_params(qz_params)
    kl = -1 / 2 * np.sum(
        np.log(np.square(np.exp(log_sigma_qz))) + 1 -
        np.square(np.exp(log_sigma_qz)) - np.square(mu_qz))
    z = sample_diag_gaussian(mu_qz, log_sigma_qz, subkey)
    ll = log_like_top_half(x, z, dec_params)
    return ll - kl

  loss_ = vmap(elbo_k, in_axes=(None, None, None, 0))(*args, *kwargs)
  return np.mean(loss_)


def optimize_params(params, train_image, seed):
  """
  Args:
    params: variational and generator model parameters
    train_image: single digit from training images.
  Return:
    Optimized phi_mu and phi_logsigma for one digit from set.
  """
  key = random.PRNGKey(seed)
  key, subkey = random.split(key)
  qz_params = init_var_params(subkey)
  grad_elbo = jit(grad(elbo_half, argnums=1))

  n = 2500
  K = 100
  lr = 0.001
  for it in tqdm(range(n)):
    key, *subs = random.split(key, K + 1)
    qz_params = qz_params + lr * grad_elbo(train_image, qz_params,
                                           params['dec'], np.stack(subs))
    if it == 0 or (it + 1) % 100 == 0:
      loss_ = elbo_half(train_image, qz_params, params['dec'], np.stack(subs))
      tqdm.write(f"Iteration {it} \t | \t ELBO {loss_:.3f}")

  return qz_params


def joint_isocountors(params, qz_params, train_image):
  """
  Args:
    params: variational (encoder) and generator network parameters
    qz_params: approximate posterior optimizer parameters

  Plot isocontours of joint distribution p(z, top half of image x) and 
  optimized approximate posterior q_phi (z | top half of image x).
  """

  def plt_isocontours(ax,
                      fn,
                      xlim=[-6, 6],
                      ylim=[-6, 6],
                      numticks=101,
                      colors=None,
                      levels=10):
    """Plot isocountours of distributions."""
    x = onp.linspace(*xlim, num=numticks)
    y = onp.linspace(*ylim, num=numticks)
    X, Y = onp.meshgrid(x, y)
    inputs = onp.concatenate(
        [onp.atleast_2d(X.ravel()),
         onp.atleast_2d(Y.ravel())])
    zs = onp.array(fn(inputs.T))
    Z = zs.reshape(X.shape)
    cs = plt.contour(X, Y, Z, colors=colors, levels=levels)
    plt.clabel(cs, inline=1, fontsize=10, fmt='%.2g')

  fig = plt.figure(figsize=(8, 8), facecolor='white')
  ax = fig.add_subplot(111, frameon=False)
  plt_isocontours(
      ax,
      lambda z: vmap(joint_ll_top_half, in_axes=(None, 0, None))
      (train_image, z, params['dec']),
      colors='g')
  plt_isocontours(
      ax,
      lambda z: vmap(diag_gaussian_log_density, in_axes=(0, None, None))
      (z, *unpack_gaussian_params(qz_params)),
      colors='b')
  plt.grid()
  plt.xlabel(r"$z_0$")
  plt.ylabel(r"$z_1$")
  lines = [
      mpl.lines.Line2D([0], [0], color='g'),
      mpl.lines.Line2D([0], [0], color='b')
  ]
  plt.title(r'Isocountours of $\log p$ and $\log q$ posteriors')
  ax.legend(lines, ['true log posterior p', 'variational log posterior q'])
  plt.tight_layout(rect=(0, 0, 1, 1))
  plt.savefig('isocountours.png')


def infer_bottom_half(params, qz_params, train_image, seed=412):
  """
  Args:
    params: decoder
    qz_params: variational optimized posterior params
    train_image: single digit trained on

  Plots original whole image beside inferred greyscale.
  """
  key = random.PRNGKey(seed)
  key, subkey = random.split(key)
  # sample z ~ approximate posterior q, feed it to decoder to find
  # Bernoulli means of p(bottom half of image | x).
  z = sample_diag_gaussian(*unpack_gaussian_params(qz_params), subkey)
  x = sigmoid(decoder(z, params['dec']))

  image_ = onp.zeros((28, 28))
  image_[:14, :] = train_image.reshape([28, 28])[:14, :]  # original top half
  image_[14:, :] = x.reshape([28, 28])[14:, :]  # inferred bottom half

  plt_im = onp.zeros((28, 28 * 2))
  plt_im[:, :28] = image_
  plt_im[:, 28:] = train_image.reshape([28, 28])

  fig, ax = plt.subplots()
  plt.imshow(plt_im, cmap=plt.cm.binary)
  plt.axis('off')
  plt.savefig('frankenstein_bottom_to_top.png', bbox_inches='tight')


if __name__ == '__main__':
  train_images, train_labels, test_images, test_labels = load_data()

  # change the seed
  seed = 412
  num_samples = 10
  train(train_images, test_images, 'params.pkl', seed)

  # plot samples form generative model
  opt_params = load_params('params.pkl')
  sample_gen(opt_params, num_samples, seed)
  latent_means(opt_params, train_images, train_labels)
  interpolate_ex = [(1, 2), (3, 8), (4, 5)]
  lin_interpolate(opt_params, train_images, train_labels, interpolate_ex)

  # non-amortized inference (we are selecting one good sample)
  select_im_good = train_images[1]
  qz_params = optimize_params(opt_params, select_im_good, seed)
  joint_isocountors(opt_params, qz_params, select_im_good)
  infer_bottom_half(opt_params, qz_params, select_im_good)