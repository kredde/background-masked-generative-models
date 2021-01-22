from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import math
from numbers import Number

class DeeperVAEVariance(LightningModule):
    def __init__(self, lr: float = 1e-3, kl_coeff: float = 0.1):
        super(DeeperVAEVariance, self).__init__()

        self.lr = lr
        self.kl_coeff = kl_coeff

        self.encoder_part = nn.Sequential(
            nn.Linear(784, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU())
        
        self.fc_mu_z = nn.Linear(64, 40)
        self.fc_log_var_z = nn.Linear(64, 40)

        self.decoder_part = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU())

        self.fc_mu_x = nn.Linear(512, 784)
        self.fc_log_var_x = nn.Linear(512, 784)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        batch_size, width, height = x.size()
        target_x = x.clone()
        target_x = torch.flatten(target_x, 1)
        
        mu_z, log_var_z = self.encode(x)
        _, _, z = self.sample_enc(mu_z, log_var_z)
        
        mu_x, log_var_x = self.decode(z.clone())

        return mu_x, log_var_x

    def encode(self, x):
        x = torch.flatten(x, 1)
        h = self.encoder_part(x)

        return self.fc_mu_z(h), self.fc_log_var_z(h)
    
    def decode(self, z):
        x_hat = self.decoder_part(z)
        return self.fc_mu_x(x_hat), self.fc_log_var_x(x_hat)

    def _run_step(self, x):
        mu_z, log_var_z = self.encode(x)
        p, q, z = self.sample_enc(mu_z, log_var_z)

        mu_x, log_var_x = self.decode(z.clone())

        return mu_x, log_var_x, p, q, z
    
    def sample_enc(self, mu_z, log_var_z):

        std = torch.exp(log_var_z / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu_z), torch.ones_like(std))
        q = torch.distributions.Normal(mu_z, std)
        z = q.rsample()
        return p, q, z
    
    def sample_dec(self, mu_x, log_var_x, target):
        mask = target.clone()
        std = torch.exp(log_var_x / 2)
        p = torch.distributions.Normal(mu_x, std)

        recons_x = p.sample()
        # recons_x = mu_x
        return recons_x
    
    def step(self, batch, batch_idx):
        x, _ = batch
        mu_x, log_var_x, p, q, z = self._run_step(x)

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
        
        mask = target.clone()
        
        # log_var = log_var * mask

        std = torch.exp(log_var / 2)
       
        var = (std ** 2)
        log_std = math.log(std) if isinstance(std, Number) else std.log()
        # log_pi = math.log(math.sqrt(2 * math.pi))

        bg_part = torch.clone((log_std))
        fg_part = torch.clone((-log_std))

        bg_part[mask > 0] = 0
        fg_part[mask == 0] = 0

        fg_log_prob = -((target - mean) ** 2) / (2 * var) + bg_part
        bg_log_prob = -((target - mean) ** 2) / (2 * var) + fg_part
        # dist = torch.distributions.Normal(mean, std)

        loss = torch.mean(-(fg_log_prob + bg_log_prob))
        return loss


