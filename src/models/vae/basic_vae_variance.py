from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from numbers import Number
import torch.nn.functional as F


class BasicVAEVariance(LightningModule):
    """
        VAE architecture used to learn variance along with the mean
    """
    def __init__(self, lr: float = 1e-3, kl_coeff: float = 0.1, use_custom_loss: bool = True, bg_aug_train: bool = False):
        super(BasicVAEVariance, self).__init__()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.use_custom_loss = use_custom_loss
        self.bg_aug_train = bg_aug_train
    
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu_z = nn.Linear(400, 80)
        self.fc_log_var_z = nn.Sequential(nn.Linear(400, 80))

        self.fc2 = nn.Linear(80, 400)
        self.fc_mu_x = nn.Linear(400, 784)
        self.fc_log_var_x = nn.Sequential(nn.Linear(400, 784))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        
        mu_z, log_var_z = self.encode(x)

        _, _, z = self.sample_enc(mu_z, log_var_z)

        mu_x, log_var_x = self.decode(z.clone())

        return mu_x, log_var_x

    def encode(self, x):
        x = torch.flatten(x, 1)
        h1 = self.relu(self.fc1(x))

        return self.fc_mu_z(h1), self.fc_log_var_z(h1)
    
    def decode(self, z):
        z = self.relu(self.fc2(z))

        return self.fc_mu_x(z), self.fc_log_var_x(z)

    def _run_step(self, x):
        mu_z, log_var_z = self.encode(x)
        p, q, z = self.sample_enc(mu_z, log_var_z)

        mu_x, log_var_x = self.decode(z.clone())

        return mu_x, log_var_x, p, q, z
    
    def sample_enc(self, mu_z, log_var_z):

        std = torch.exp(log_var_z/2)
        p = torch.distributions.Normal(torch.zeros_like(mu_z), torch.ones_like(std))
        q = torch.distributions.Normal(mu_z, std)
        z = q.rsample()
        return p, q, z
    
    def sample_dec(self, mu_x, log_var_x, target):
        std = torch.exp(log_var_x/2)
        p = torch.distributions.Normal(mu_x, std)

        recons_x = p.sample()
        return recons_x

    def step(self, batch, batch_idx):
        if self.bg_aug_train:
            x, mask, _ = batch
        else:
            x, _ = batch
            mask = x.clone()

        mu_x, log_var_x, p, q, z = self._run_step(x)
       
        x = torch.flatten(x, 1)
        mask = torch.flatten(mask, 1)

        recon_loss = self.loss_func(x, mask, mu_x, log_var_x)

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


    def loss_func(self, target, mask, mean, log_var):
        
        std = torch.exp(log_var/2)
        
        var = (std ** 2)
        log_std = math.log(std) if isinstance(std, Number) else std.log()

        bg_part = torch.clone((log_std))
        full_img = torch.clone((-log_std))
            
        bg_part[mask > 0.0] = 0.0
        if self.use_custom_loss:
            log_prob = -((target - mean) ** 2) / (2 * var) + full_img + bg_part
        else:
            log_prob = -((target - mean) ** 2) / (2 * var) + full_img

        loss = torch.mean(-(log_prob))
        return loss

