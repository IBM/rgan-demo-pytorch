import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from rgan.rgan import Generator, Discriminator


class NoiseGenerator(nn.Module):
    def __init__(self, noise_scale):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.Tensor(noise_scale))
    

class rGAN_noise_variable(pl.LightningModule):

    def __init__(self, mm, x_dataset, y_dataset, hparams, 
                 G, R, D_X, D_Y, D_noise, G_noise,
                 z_dim, train_stage='rgan'):
        super(rGAN_noise_variable, self).__init__()
        # mm is the mechanistic model
        self.mm = mm
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.save_hyperparameters(hparams)
        self.G = G
        self.R = R
        self.D_X = D_X
        self.D_Y = D_Y
        self.D_noise = D_noise
        self.G_noise = G_noise
        self.z_dim = z_dim
        self.train_stage = train_stage

    def prior_state_dict(self):
        state = {
            'G': self.G.state_dict(),
            'D_X': self.D_X.state_dict(),
            'D_noise': self.D_noise.state_dict(),
            'R': self.R.state_dict(),
            'G_noise': self.G_noise.state_dict()
        }
        return state
        
    def load_prior_state_dict(self, state):
        self.G.load_state_dict(state['G'])
        self.D_X.load_state_dict(state['D_X'])
        self.D_noise.load_state_dict(state['D_noise'])
        self.R.load_state_dict(state['R'])
        self.G_noise.load_state_dict(state['G_noise'])
        
    def train_dataloader(self):
        loader_x = DataLoader(self.x_dataset, batch_size=self.hparams.batch_size)
        loader_y = DataLoader(self.y_dataset, batch_size=self.hparams.batch_size)
        return {'x': loader_x, 'y': loader_y}

    # Loss functions
    def reconstruction_loss(self, x, y):
        return nn.MSELoss()(x, y)
    
    def generator_loss(self, gen_cls):
        return (torch.log(1-gen_cls) - torch.log(gen_cls)).mean()
    
    def discriminator_loss(self, gen_cls, true_cls):
        return (-torch.log(true_cls) - torch.log(1.0 - gen_cls)).mean()

    # Optimizers
    def configure_optimizers(self):
        betas = (0.9, 0.999)
        optimizer_G = torch.optim.Adam([{'params': self.G.parameters()}, 
                                        {'params': self.R.parameters()},
                                        {'params': self.G_noise.parameters()}],
                                       lr=self.hparams.lr_g,
                                       betas=betas)
        optimizer_D_X = torch.optim.Adam(self.D_X.parameters(), 
                                         lr=self.hparams.lr_d_x,
                                         betas=betas)
        optimizer_D_Y = torch.optim.Adam(self.D_Y.parameters(), 
                                         lr=self.hparams.lr_d_y,
                                         betas=betas)
        optimizer_D_noise = torch.optim.Adam(self.D_noise.parameters(), 
                                             lr=self.hparams.lr_d_n,
                                             betas=betas)
        # return the list of optimizers and second empty list is for schedulers (if any)
        # Order of the optimizers matters - have to generate samples first when 
        # optimizing optimizer_G
        return [optimizer_G, optimizer_D_X, optimizer_D_Y, optimizer_D_noise], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        data_x = batch['x']
        data_y = batch['y']

        # sample base variable
        z = torch.randn(data_x.shape[0], self.z_dim).type_as(data_x)

        # saving loss_function as local variable
        g_criterion = self.generator_loss
        d_criterion = self.discriminator_loss
        r_criterion = self.reconstruction_loss

        # Generator training
        if optimizer_idx == 0:
            # Generate samples
            self.x_g = self.G(z)

            # sample noise
            self.z_noise = torch.randn(data_x.shape[0], self.z_dim).type_as(data_x)
            self.noise_g = self.z_noise * self.G_noise.noise_scale
            self.noise_g_n = self.mm.noise_N(self.noise_g)
            
            # Generator loss
            g_loss_x = g_criterion(self.D_X(self.x_g))

            # Noise loss (should only contribute a gradient for noise_scale variable)
            g_loss_noise = g_criterion(self.D_noise(self.noise_g_n))

            # Model output discriminator loss
            if self.train_stage == 'rgan':
                self.y_g = self.mm(torch.concat([self.x_g, self.noise_g_n], dim=1))
                g_loss_y = g_criterion(self.D_Y(self.y_g))
            elif self.train_stage == 'prior':
                g_loss_y = 0.0 
            
            # Reconstruction Loss
            g_loss_r = r_criterion(z, self.R(self.x_g))
            
            g_loss = g_loss_x * self.hparams.wx + \
                     g_loss_y * self.hparams.wy + \
                     g_loss_r * self.hparams.wr + \
                     g_loss_noise * self.hparams.wn
            
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss
        
        # D_X training
        if optimizer_idx == 1:
            # # D_X loss
            d_x_loss = d_criterion(self.D_X(self.x_g.detach()), self.D_X(data_x))
        
            self.log("d_x_loss", d_x_loss, prog_bar=True)
            return d_x_loss

        # D_Y training
        if optimizer_idx == 2:
            # D_Y loss
            if self.train_stage == 'rgan':
                d_y_loss = d_criterion(self.D_Y(self.y_g.detach()), self.D_Y(data_y))
            elif self.train_stage == 'prior':
                return None

            self.log("d_y_loss", d_y_loss, prog_bar=True)
            return d_y_loss
        
        # D_noise training
        if optimizer_idx == 3:
            # # D_noise loss
            d_n_loss = d_criterion(
                self.D_noise(self.noise_g_n.detach()), 
                self.D_noise(self.z_noise.detach())
            )
        
            self.log("d_n_loss", d_n_loss, prog_bar=True)
            return d_n_loss


def defaultNetsRGAN_noise_variable(z_dim=2, x_dim=2, y_dim=1, noise_dim=2,
                                   g_layers=8, g_nodes=100,
                                   x_d_layers=8, x_d_nodes=100,
                                   y_d_layers=8, y_d_nodes=100,
                                   noise_scale=[0.0825, 0.0825]):
    G = Generator(z_dim=z_dim,
                 x_dim=x_dim,
                 n_hidden_layers=g_layers,
                 n_hidden_nodes=g_nodes)
    R = Generator(z_dim=x_dim,
                  x_dim=z_dim,
                  n_hidden_layers=g_layers,
                  n_hidden_nodes=g_nodes)
    D_X = Discriminator(dim=x_dim,
                        n_hidden_layers=x_d_layers,
                        n_hidden_nodes=x_d_nodes,
                        dropout=0.01, specnorm=None)
    D_Y = Discriminator(dim=y_dim,
                        n_hidden_layers=y_d_layers,
                        n_hidden_nodes=y_d_nodes,
                        dropout=None, specnorm=True)
    D_N = Discriminator(dim=noise_dim,
                        n_hidden_layers=x_d_layers,
                        n_hidden_nodes=x_d_nodes,
                        dropout=0.01, specnorm=False)
    G_N = NoiseGenerator(noise_scale=noise_scale)

    return G, R, D_X, D_Y, D_N, G_N
