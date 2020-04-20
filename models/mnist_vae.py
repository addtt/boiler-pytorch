import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from boilr.models import BaseGenerativeModel

data_std = .02

class MnistVAE(BaseGenerativeModel):

    def __init__(self, z_dim=8):
        super().__init__()
        self.z_dim = z_dim
        self.pz = Normal(0., 1.)
        self.encoder = nn.Sequential(
            nn.Linear(28**2, 256),
            nn.LeakyReLU(),
            nn.Dropout(.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(.2),
            nn.Linear(64, 2 * z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(.2),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Dropout(.2),
            nn.Linear(256, 28**2),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        mu, lv = torch.chunk(self.encoder(x_flat), 2, dim=1)
        qz = Normal(mu, (lv / 2).exp())
        z = qz.rsample()
        mean = self.decoder(z).view_as(x)
        pxz = Normal(mean, data_std)
        nll = -pxz.log_prob(x).sum((1, 2, 3))
        kl = kl_divergence(qz, self.pz).sum(1)
        elbo = -nll - kl
        return {
            'mean': mean,
            'sample': pxz.sample(),
            'nll': nll,
            'kl': kl,
            'elbo': elbo,
        }


    def sample_prior(self, n_imgs, **kwargs):
        z = self.pz.sample([n_imgs, self.z_dim]).to(self.get_device())
        mean = self.decoder(z).view((n_imgs, 1, 28, 28))
        pxz = Normal(mean, data_std)
        return pxz.sample()
