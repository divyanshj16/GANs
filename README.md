# GANs
This repository is a comparitive study of [Vanilla GAN](https://arxiv.org/abs/1406.2661), [LSGAN](https://arxiv.org/abs/1611.04076), [DCGAN](https://arxiv.org/abs/1511.06434), and [WGAN](https://arxiv.org/abs/1704.00028) for MNIST dataset on both tensorflow and pytorch frameworks.

## What is this GAN thing?

In 2014, [Goodfellow et al.](https://arxiv.org/abs/1406.2661) presented a method for training generative models called Generative Adversarial Networks (GANs for short). In a GAN, there are two different neural networks. Our first network is a traditional classification network, called the **discriminator**. We will train the discriminator to take images, and classify them as being real (belonging to the training set) or fake (not present in the training set). Our other network, called the **generator**, will take random noise as input and transform it using a neural network to produce images. The goal of the generator is to fool the discriminator into thinking the images it produced are real.

![GAN](https://cdn-images-1.medium.com/max/1000/1*-gFsbymY9oJUQJ-A3GTfeg.png)

Since 2014, GANs have exploded into a huge research area, with massive [workshops](https://sites.google.com/site/nips2016adversarial/), and [hundreds of new papers](https://github.com/hindupuravinash/the-gan-zoo). Compared to other approaches for generative models, they often produce the highest quality samples but are some of the most difficult and finicky models to train (see [this github repo](https://github.com/soumith/ganhacks) that contains a set of 17 hacks that are useful for getting models working). Improving the stabiilty and robustness of GAN training is an open research question, with new papers coming out every day! For a more recent tutorial on GANs, see [here](https://arxiv.org/abs/1701.00160). There is also some even more recent exciting work that changes the objective function to Wasserstein distance and yields much more stable results across model architectures: [WGAN](https://arxiv.org/abs/1701.07875), [WGAN-GP](https://arxiv.org/abs/1704.00028).

GANs are not the only way to train a generative model! For other approaches to generative modeling check out the [deep generative model chapter](http://www.deeplearningbook.org/contents/generative_models.html) of the Deep Learning [book](http://www.deeplearningbook.org). Another popular way of training neural networks as generative models is Variational Autoencoders (co-discovered [here](https://arxiv.org/abs/1312.6114) and [here](https://arxiv.org/abs/1401.4082)). Variational autoencoders combine neural networks with variational inference to train deep generative models. These models tend to be far more stable and easier to train but currently don't produce samples that are as pretty as GANs.

## Model Architecture

![Architechture of GAN](https://deeplearning4j.org/img/GANs.png)

Both Generator and Discriminator are stacked fully-connected layers in case of **Vanilla GAN** and Convolutional in case of **DCGAN**

## Loss Function

### Vanilla GAN

This loss function is from the original paper by [Goodfellow et al.](https://arxiv.org/abs/1406.2661). It can be thought as minimax game between generator and discriminator where generator ($G$) trying to fool the discriminator ($D$), and the discriminator trying to correctly classify real vs. fake.

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\underset{G}{\text{minimize}}\;&space;\underset{D}{\text{maximize}}\;&space;\mathbb{E}_{x&space;\sim&space;p_\text{data}}\left[\log&space;D(x)\right]&space;&plus;&space;\mathbb{E}_{z&space;\sim&space;p(z)}\left[\log&space;\left(1-D(G(z))\right)\right]$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\underset{G}{\text{minimize}}\;&space;\underset{D}{\text{maximize}}\;&space;\mathbb{E}_{x&space;\sim&space;p_\text{data}}\left[\log&space;D(x)\right]&space;&plus;&space;\mathbb{E}_{z&space;\sim&space;p(z)}\left[\log&space;\left(1-D(G(z))\right)\right]$$" title="$$\underset{G}{\text{minimize}}\; \underset{D}{\text{maximize}}\; \mathbb{E}_{x \sim p_\text{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p(z)}\left[\log \left(1-D(G(z))\right)\right]$$" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=$x&space;\sim&space;p_\text{data}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x&space;\sim&space;p_\text{data}$" title="$x \sim p_\text{data}$" /></a> are samples from the input data, $z \sim p(z)$ are the random noise samples, $G(z)$ are the generated images using the neural network generator $G$, and $D$ is the output of the discriminator, specifying the probability of an input being real. In [Goodfellow et al.](https://arxiv.org/abs/1406.2661), they analyze this minimax game and show how it relates to minimizing the Jensen-Shannon divergence between the training data distribution and the generated samples from $G$.

### LSGAN

### WGAN

## How to run?

## Dependencies

## System Requirements

## Results
