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


class rGAN(pl.LightningModule):

    def __init__(self, mm, x_dataset, y_dataset, hparams, 
                 G, R, D_X, D_Y, z_dim, train_stage='rgan'):
        super(rGAN, self).__init__()
        # mm is the mechanistic model
        self.mm = mm
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.save_hyperparameters(hparams)
        self.G = G
        self.R = R
        self.D_X = D_X
        self.D_Y = D_Y
        self.z_dim = z_dim
        self.train_stage = train_stage

    def prior_state_dict(self):
        state = {
            'G': self.G.state_dict(),
            'D_X': self.D_X.state_dict(),
            'R': self.R.state_dict()
        }
        return state
        
    def load_prior_state_dict(self, state):
        self.G.load_state_dict(state['G'])
        self.D_X.load_state_dict(state['D_X'])
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
        optimizer_D_X = torch.optim.Adam(self.D_X.parameters(), 
                                         lr=self.hparams.lr_d_x,
                                         betas=betas)
        optimizer_D_Y = torch.optim.Adam(self.D_Y.parameters(), 
                                         lr=self.hparams.lr_d_y,
                                         betas=betas)
        # return the list of optimizers and second empty list is for schedulers (if any)
        # Order of the optimizers matters - have to generate samples first when 
        # optimizing optimizer_G
        return [optimizer_G, optimizer_D_X, optimizer_D_Y], []


    def training_step(self, batch, batch_idx, optimizer_idx):
        data_x = batch['x']
        data_y = batch['y']

        # sample noise
        z = torch.randn(data_x.shape[0], self.z_dim)
        z = z.type_as(data_x)

        # saving loss_function as local variable
        g_criterion = self.generator_loss
        d_criterion = self.discriminator_loss
        r_criterion = self.reconstruction_loss

        # Generator training
        if optimizer_idx == 0:
            # Generate samples
            self.x_g = self.G(z)

            # Generator loss
            g_loss_x = g_criterion(self.D_X(self.x_g))
       
            if self.train_stage == 'rgan':
                self.y_g = self.mm(self.x_g)
                g_loss_y = g_criterion(self.D_Y(self.y_g))
            elif self.train_stage == 'prior':
                g_loss_y = 0.0
               
            g_loss_r = r_criterion(z, self.R(self.x_g))
            
            g_loss = g_loss_x * self.hparams.wx + \
                     g_loss_y * self.hparams.wy + \
                     g_loss_r * self.hparams.wr
            
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


def defaultNetsRGAN(z_dim=2, x_dim=2, y_dim=1, 
                    g_layers=8, g_nodes=100,
                    x_d_layers=8, x_d_nodes=100,
                    y_d_layers=8, y_d_nodes=100):
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

    return G, R, D_X, D_Y
