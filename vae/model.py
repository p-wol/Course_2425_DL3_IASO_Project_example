import torch
from torch import nn

class VAE_FC(nn.Module):
    def __init__(self, layers, latent_dim = 200, leak = .1):
        super(VAE_FC, self).__init__()
        
        self.layers = layers
        self.latent_dim = latent_dim
        
        # encoder
        encoder = []
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            encoder += [nn.Linear(l_in, l_out), nn.LeakyReLU(leak)]

        # encoder
        self.encoder = nn.Sequential(*encoder)
        
        # latent mean and variance
        self.fc_mean = nn.Linear(layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(layers[-1], latent_dim)
        
        # decoder input
        self.decoder_input = nn.Linear(latent_dim, layers[-1])
        
        # decoder
        rev_layers = list(reversed(layers))
        decoder = [nn.LeakyReLU(leak)]
        for l_in, l_out in zip(rev_layers[:-2], rev_layers[1:-1]):
            decoder += [nn.Linear(l_in, l_out), nn.LeakyReLU(leak)]
        decoder += [nn.Linear(layers[1], layers[0]), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)
     
    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        mean, logvar = self.fc_mean(x), self.fc_logvar(x)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        device = self.fc_mean.weight.device
        epsilon = torch.randn_like(logvar).to(device)
        z = mean + (.5*logvar).exp() * epsilon
        return z

    def decode(self, x):
        x = self.decoder_input(x)
        x = self.decoder(x)
        return x.view(x.size(0), 1, 28, 28)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    
    def sample(self, num_samples):
        device = self.fc_mean.weight.device
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device = device)
            samples = self.decode(z)

            return samples
    
    def reconstruct(self, x):
        with torch.no_grad():
            mean, logvar = self.encode(x)
            return self.decode(mean)
