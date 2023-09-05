import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class ResBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        padding = kernel_size//2
        output_padding = stride-1
        conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding)
        batchnorm = nn.BatchNorm2d(out_channels)
        activation = nn.ReLU()

        self.block = nn.Sequential(
            conv,
            batchnorm,
            activation
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, 
                                   stride=stride, padding=0, 
                                   output_padding=output_padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
   
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        oout = out + residual
        return oout


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, 
                 stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size//2
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding)
        dropout = nn.Dropout(dropout)
        activation = nn.ReLU()

        self.block = nn.Sequential(
            conv,
            dropout,
            activation
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                          kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = None
   
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        oout = out + residual
        return oout


class Generator(nn.Module):
    def __init__(self, z_dim, x_shape):
        super().__init__()
        self.img_shape = x_shape
        self.init_size = self.img_shape[1] // 4
        linear_dim = 256 * self.init_size ** 2

        self.linear_layers = [nn.Linear(z_dim, linear_dim),
                              nn.BatchNorm1d(linear_dim),
                              nn.LeakyReLU()]

        self.model = nn.Sequential(
            *self.linear_layers,
            nn.Unflatten(1, (256, self.init_size, self.init_size)),
            ResBlockTranspose(256, 128),
            ResBlockTranspose(128, 128),
            ResBlockTranspose(128, 64, stride=2),
            ResBlockTranspose(64, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, 
                               padding=2, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.model(z)
        return x
    

class ConvResNet(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.model = nn.Sequential(
            ResBlock(1, 64, stride=2, dropout=dropout),
            ResBlock(64, 64, dropout=dropout),
            ResBlock(64, 128, stride=2, dropout=dropout),
            ResBlock(128, 128, dropout=dropout),
        )

    def forward(self, x):
        out = self.model(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.model = nn.Sequential(
            ConvResNet(dropout=0.1),
            nn.Flatten(),
            nn.Linear((-(-self.dim//4))**2*128, 1),
            nn.Sigmoid(),
        )

    def forward(self, y):        
        out = self.model(y)
        return out
    

class Reconstructor(nn.Module):
    def __init__(self, x_dim, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        self.model = nn.Sequential(
            ConvResNet(dropout=0.4),
            nn.Flatten(),
            nn.Linear((-(-x_dim//4))**2*128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.z_dim),
        )

    def forward(self, x):        
        z = self.model(x)
        return z


class rGAN(pl.LightningModule):

    def __init__(self, mm, x_dataset, y_dataset, hparams, 
                 G, R, D_X, D_Y, z_dim, train_stage='rgan'):
        super(rGAN, self).__init__()
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
                g_loss_y = 0.0 # torch.zeros((1,))
               
            g_loss_r = r_criterion(z, self.R(self.x_g))
            
            g_loss = g_loss_x * self.hparams.wx + \
                     g_loss_y * self.hparams.wy + \
                     g_loss_r * self.hparams.wr
            
            # self.log("g_loss", g_loss, prog_bar=True)
            self.log_dict({
                "g": g_loss,
                "g_x": g_loss_x,
                "g_y": g_loss_y,
                "g_r": g_loss_r,
            }, prog_bar=True)
            
            return g_loss
        
        # D_X training
        if optimizer_idx == 1:
            # # D_X loss
            true = self.D_X(data_x)
            gen = self.D_X(self.x_g.detach())
            d_x_loss = d_criterion(gen, true)
            # d_x_loss = d_criterion(self.D_X(self.x_g.detach()), self.D_X(data_x))
            
        
            # self.log("d_x_loss", d_x_loss, prog_bar=True)
            self.log_dict({
                "d_x": d_x_loss,
                "d_x_t": true.mean(),
                "d_x_g": gen.mean(),
            }, prog_bar=True)
            return d_x_loss

        # D_Y training
        if optimizer_idx == 2:
            # D_Y loss
            if self.train_stage == 'rgan':
                true = self.D_Y(data_y)
                gen = self.D_Y(self.y_g.detach())
                d_y_loss = d_criterion(gen, true)
                # d_y_loss = d_criterion(self.D_Y(self.y_g.detach()), self.D_Y(data_y))
                self.log_dict({
                    "d_y": d_y_loss,
                    "d_y_t": true.mean(),
                    "d_y_g": gen.mean(),
                }, prog_bar=True)
            elif self.train_stage == 'prior':
                return None

            # self.log("d_y_loss", d_y_loss, prog_bar=True)
            return d_y_loss


def MNISTNetsRGAN(z_dim=100, x_shape=(28,28), y_shape=(22,22)):
    G = Generator(z_dim=z_dim,
                  x_shape=x_shape)
    R = Reconstructor(x_dim=x_shape[0], z_dim=z_dim)
    D_X = Discriminator(dim=x_shape[0])
    D_Y = Discriminator(dim=y_shape[0])
    return G, R, D_X, D_Y
