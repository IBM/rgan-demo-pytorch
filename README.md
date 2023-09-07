[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IBM/rgan-demo-pytorch/main?labpath=examples/nonlinear/uniform_prior.ipynb)

This repository provides pytorch source code for demonstrations associated with our _(DOI)_ preprint publication, "Novel and flexible parameter estimation methods for data-consistent inversion in mechanistic modeling".

<!-- Paper: [Arxiv Link](https://arxiv.org/) -->

# Regularized Generative Adversarial Network (r-GAN)

**r-GAN** is a generative model for solving a stochastic inverse problem (SIP). The goal is to generate a distribution of mechanistic model (MM) parameters that, when supplied as input to the MM to produce a distribution of MM outputs, matches a distribution of real observations in the MM output domain. 

## Demo notebooks

Demonstrations that follow the examples in section 3 of the manuscript can be found in [examples](examples). For each example problem, we have provided solutions using r-GAN, modified rejection sampling (algorithm 2 in the manuscript), and c-GAN.

The included examples are:

* [Nonlinear function, uniform prior](examples/nonlinear/uniform_prior.ipynb) (Figure 2A-G)
* [Nonlinear function, beta prior](examples/nonlinear/beta_prior.ipynb) (Figure 2H-N)
* ["Wobbly Plate", deterministic](examples/wobbly_plate/deterministic.ipynb) (Figure 3A-G)
* ["Wobbly Plate", stochastic](examples/wobbly_plate/stochastic.ipynb) (Figure 3H-O)
* [MNIST super-resolution imaging](examples/MNIST/rgan_mnist_demo_full.ipynb) (Figure 4)


# Python Environment 

## Conda Create and Activate Environment Manually

Either install from environment file:
```
conda env create -f environment.yml
conda activate rgan_pytorch
```

Or use requirements file with pip, or create manually:


### Conda manual environment creation


```
conda create --name rgan_pytorch python=3.10.8
conda activate rgan_pytorch
```

### Conda Install Packages
```
conda install jupyter==1.0.0 numpy==1.23.5 matplotlib==3.6.2 seaborn==0.12.2 scipy==1.11.1 scikit-learn==1.3.0
conda install pytorch==1.13.1 -c pytorch
conda install pytorch-lightning==1.9.3 -c conda-forge 
```

Additional package installation is required to enable CUDA.

## Timings

Approximate duration of demo stages (using 2018 MacBook Pro with 6 core i7 CPU)

Environment install time: ~1 minute

r-GAN Prior training stage: ~6 minutes

r-GAN training stage: ~17 minutes

Approximate duration of MNIST demo stages (using a single NVIDIA V100 GPU (additional pytorch setup to use CUDA is required))

Environment install time: ~1 minute

r-GAN Prior training stage: ~15 minutes

r-GAN training stage: ~30 minutes
