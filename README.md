This repository provides pytorch source code for a demonstration associated with our _(DOI)_ preprint publication, "Parameter estimation for biological model ensembles with data-consistent inversion".

Paper: [Arxiv Link](https://arxiv.org/)

# Regularized Generative Adversarial Network (r-GAN)

**r-GAN** is a generative model for solving a stochastic inverse problem (SIP). The goal is to generate a distribution of mechanistic model (MM) parameters that, when supplied as input to the MM to produce a distribution of MM outputs, matches a distribution of real observations in the MM output domain. 

## Demo notebook

A demonstration of the r-GAN that follows the example in section 3A of the manuscript (Figure 2A-G) can be found in [this notebook](test.ipynb).

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
conda install jupyter==1.0.0 numpy==1.23.5 matplotlib==3.6.2 seaborn==0.12.2
conda install pytorch==1.13.1 -c pytorch
conda install pytorch-lightning==1.9.3 -c conda-forge 
```

## Timings

Approximate duration of demo stages (using 2018 MacBook Pro with 6 core i7 CPU)

Environment install time: ~1 minute

r-GAN Prior training stage: ~6 minutes

r-GAN training stage: ~17 minutes
