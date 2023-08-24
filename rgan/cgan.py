import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, n_hidden_layers=8, n_hidden_nodes=100):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes

        def layer(in_dim, out_dim):
            layers = [nn.Linear(in_dim, out_dim)]
            layers.append(nn.ReLU())
            return layers
        
        def hidden_layers(n_layers):
            layers = []
            for l in range(n_layers):
                layers.extend(layer(self.n_hidden_nodes, self.n_hidden_nodes))
            return layers

        self.model = nn.Sequential(
            *layer(z_dim, self.n_hidden_nodes),
            *hidden_layers(self.n_hidden_layers),
            nn.Linear(self.n_hidden_nodes, self.x_dim),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, dim, n_hidden_layers=8, n_hidden_nodes=100,
                 dropout=None, specnorm=None):
        super().__init__()
        self.dim = dim
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.dropout = dropout
        self.specnorm = specnorm

        def layer(in_dim, out_dim):
            if self.specnorm:
                layers = [spectral_norm(nn.Linear(in_dim, out_dim))]
            else:
                layers = [nn.Linear(in_dim, out_dim)]
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=dropout))

            return layers
        
        def hidden_layers(n_layers):
            layers = []
            for l in range(n_layers):
                layers.extend(layer(self.n_hidden_nodes, self.n_hidden_nodes))
            return layers

        self.model = nn.Sequential(
            *layer(dim, self.n_hidden_nodes),
            *hidden_layers(self.n_hidden_layers),
            nn.Linear(self.n_hidden_nodes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class cGAN(pl.LightningModule):

    def __init__(self, x_dataset, y_dataset, hparams, 
                 G, R, D, z_dim):
        super(cGAN, self).__init__()

        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.save_hyperparameters(hparams)
        self.G = G
        self.R = R
        self.D = D
        self.z_dim = z_dim

    def prior_state_dict(self):
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'R': self.R.state_dict()
        }
        return state
        
    def load_prior_state_dict(self, state):
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])
        self.R.load_state_dict(state['R'])
        
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
                                        {'params': self.R.parameters()}],
                                       lr=self.hparams.lr_g,
                                       betas=betas)
        optimizer_D = torch.optim.Adam(self.D.parameters(), 
                                       lr=self.hparams.lr_d,
                                       betas=betas)
        # return the list of optimizers and second empty list is for schedulers (if any)
        # Order of the optimizers matters - have to generate samples first when 
        # optimizing optimizer_G
        return [optimizer_G, optimizer_D], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        data_x = batch['x']
        data_y = batch['y']

        # sample noise
        z = torch.randn(data_x.shape[0], self.z_dim)
        z = z.type_as(data_x)

        # Input to G
        G_input = torch.concat([z, data_y], dim=1)

        # Target in D
        D_target = torch.concat([data_y, data_x], dim=1)

        # saving loss_function as local variable
        g_criterion = self.generator_loss
        d_criterion = self.discriminator_loss
        r_criterion = self.reconstruction_loss

        # Generator training
        if optimizer_idx == 0:
            # Generate samples
            self.x_g = self.G(G_input)

            # Generator loss
            self.D_generated = torch.concat([data_y, self.x_g], dim=1)
            g_loss_x = g_criterion(self.D(self.D_generated))
               
            g_loss_r = r_criterion(z, self.R(self.x_g))
            
            g_loss = g_loss_x * self.hparams.wx + \
                     g_loss_r * self.hparams.wr
            
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss
        
        # D training
        if optimizer_idx == 1:
            # # D_X loss
            d_loss = d_criterion(self.D(self.D_generated.detach()), self.D(D_target))
        
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss


def defaultNetsCGAN(z_dim=2, x_dim=2, y_dim=1, 
                    g_layers=8, g_nodes=100,
                    d_layers=8, d_nodes=100):
    G = Generator(z_dim=z_dim+y_dim,
                  x_dim=x_dim,
                  n_hidden_layers=g_layers,
                  n_hidden_nodes=g_nodes)
    R = Generator(z_dim=x_dim,
                  x_dim=z_dim,
                  n_hidden_layers=g_layers,
                  n_hidden_nodes=g_nodes)
    D = Discriminator(dim=x_dim+y_dim,
                      n_hidden_layers=d_layers,
                      n_hidden_nodes=d_nodes,
                      dropout=0.01, specnorm=None)

    return G, R, D
