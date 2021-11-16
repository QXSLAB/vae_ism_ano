import torch
import torch.nn as nn


def basic_conv(inc, out_c):

    block = nn.Sequential(
        nn.Conv2d(inc, out_c, 4, 2, 1),
        #nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2)
    )

    return block


def basic_deconv(inc, out_c):

    block = nn.Sequential(
        nn.ConvTranspose2d(inc, out_c, 4, 2, 1),
        #nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2)
    )

    return block


class VAE(nn.Module):

    def __init__(self):

        super().__init__()

        ch = 32
        depth = 4
        bottle = 75
        flat = ch * (2**depth)
        self.flat = flat
        
        self.encoder = nn.Sequential(
            *[basic_conv(ch*2**(i-1) if i>1 else 1, ch*2**i)
                for i in range(1, depth+1)],
            nn.Conv2d(flat, flat, 4, 4, 0)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(flat, 4096),
            nn.LeakyReLU(0.2),
        )
        
        self.to_mean = nn.Sequential(
            nn.Linear(4096, bottle),
        )

        self.to_log = nn.Sequential(
            nn.Linear(4096, bottle),
        )

        self.from_bottle = nn.Sequential(
            nn.Linear(bottle, flat),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(flat, flat, 4, 4, 0),
            *[basic_deconv(ch*2**i, ch*2**(i-1))
                for i in range(depth, 0, -1)],
            nn.Conv2d(ch, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
    def resample(self, mean, logvar):

        std = logvar.mul(0.5).exp()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add(mean)
    
    def forward(self, x):

        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        mean = self.to_mean(x)
        logvar = self.to_log(x)
        z = self.resample(mean, logvar)

        y = self.from_bottle(z)
        y = y.reshape(-1, self.flat, 1, 1)
        y = self.decoder(y)

        return y, mean, logvar


def basic_fc(inc, out_c):

    block = nn.Sequential(
        nn.Linear(inc, out_c),
        nn.LeakyReLU(0.2)
    )

    return block


class VAE_FC(nn.Module):

    def __init__(self):

        super().__init__()

        size = 64
        bottle = 75
        factor = 2
        self.size = size
        
        self.encoder = nn.Sequential(
            basic_fc(size**2, 1024*factor),
            basic_fc(1024*factor, 512*factor),
            basic_fc(512*factor, 256*factor),
            basic_fc(256*factor, 128*factor),
        )

        
        self.to_mean = nn.Sequential(
            nn.Linear(128*factor, bottle),
        )

        self.to_log = nn.Sequential(
            nn.Linear(128*factor, bottle),
        )

        self.from_bottle = nn.Sequential(
            nn.Linear(bottle, 128*factor),
        )

        self.decoder = nn.Sequential(
            basic_fc(128*factor, 256*factor),
            basic_fc(256*factor, 512*factor),
            basic_fc(512*factor, 1024*factor),
            basic_fc(1024*factor, size**2)
        )

        
    def resample(self, mean, logvar):

        std = logvar.mul(0.5).exp()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add(mean)
    
    def forward(self, x):

        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)

        mean = self.to_mean(x)
        logvar = self.to_log(x)
        z = self.resample(mean, logvar)

        y = self.from_bottle(z)
        y = self.decoder(y)
        y = y.reshape(-1, 1, self.size, self.size)

        return y, mean, logvar


def vae_loss(y, mean, logvar, x):

    # TODO tune this hypermeter
    lmd = 0.1
    mse = ((y-x)**2).sum()
    kld = 0.5*(1+logvar-mean**2-logvar.exp()).sum()
    return mse - lmd*kld

