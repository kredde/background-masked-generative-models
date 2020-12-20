import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.vae.vae import VAE
from src.models.vae.components import resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder
import math


class VAEVariance(LightningModule):
    def __init__(
        self,
        input_dim: tuple = (32, 32),
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        background_only: bool = True,
        dec_out_dim: int = 3072,
        output_dim: int = 3072,
        learn_variance: bool = True,
        epsilon: float = 1e-4,
        *args, 
        **kwargs):

        super(VAEVariance, self).__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.background_only = background_only

        self.dec_out_dim = dec_out_dim
        self.output_dim = output_dim
        self.learn_variance = learn_variance
        self.epsilon = epsilon

        valid_encoders = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_dim[0], first_conv, maxpool1, learn_variance=learn_variance)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_dim[0], first_conv, maxpool1, learn_variance=learn_variance)
        
        self.fc_mu = nn.Sequential(nn.Linear(self.enc_out_dim, self.latent_dim), nn.Softplus())
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim) 
            
        self.fc_mu_x = nn.Linear(self.dec_out_dim, self.output_dim) 
        self.fc_var_x = nn.Sequential(nn.Linear(self.dec_out_dim, self.output_dim), nn.Softplus())
            
    
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        target_x = x.clone()
        target_x = torch.flatten(target_x, 1)

        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        p, q, z = self.sample(mu, log_var)
        
        z = self.decoder(z)
        mu_x = self.fc_mu_x(z)
        log_var_x = self.fc_var_x(z)

        return torch.reshape(self._sample_dec(mu_x, log_var_x, target_x), (batch_size, channels, width, height))
    
    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x) 
        p, q, z = self.sample(mu, log_var)

        z_x = self.decoder(z)
        mu_x = self.fc_mu_x(z_x)
        log_var_x = self.fc_var_x(z_x)

        return z, mu_x, log_var_x, p, q
    
    def sample(self, mu, log_var):

        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z
    
    def _sample_dec(self, mu_hat, log_var_hat, x):
        mask = x.clone()

        # mask[mask > 0] = 0.03 #reduce std at foreground pixels
        # mask[mask == 0] = 1 #keep std at background pixels as it is

        std = torch.exp(log_var_hat / 2)
        print(f'background variance ====> {log_var_hat[mask == 0]}')
        print(f'foreground variance ====> {log_var_hat[mask > 0]}')

        print(f'background std =====> {std[mask == 0]}')
        print(f'foreground std =====> {std[mask > 0]}')

        import sys
        sys.exit(0)
        # # identity_std = torch.ones_like(std) * 0.03
        # std = std * mask

        q = torch.distributions.Normal(mu_hat, std)
        
        # z = q.sample()
        z = mu_hat
        return z
    
    def step(self, batch, batch_idx):
        if self.background_only:
            x, _ = batch
        else:
            x, _ = batch[0]
            x_mask, _ = batch[1]
        
        x = x.repeat(1, 3, 1, 1)

        batch_size, channels, width, height = x.size()

        z, mu_x, log_var_x, p, q = self._run_step(x)

        x = torch.flatten(x, 1)
       
        recon_loss = self.loss_func(x, mu_x, log_var_x)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss_func(self, target, mean, log_var):
        # mse = (mean - target) ** 2
        # loss = torch.mean(log_var + (1/2 * 1/(torch.exp(log_var)) * mse))

        # return loss
        mask = target.clone()
        mask[mask > 0] = 0.1 #reduce std at foreground pixels
        mask[mask == 0] = 0.5 #keep std at background pixels as it is

        log_var = log_var * mask

        std = torch.exp(log_var / 2)

        dist = torch.distributions.Normal(mean, std)

        loss = torch.mean(-dist.log_prob(target))
        return loss



