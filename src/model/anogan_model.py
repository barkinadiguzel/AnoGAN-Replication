import torch
from src.generator.g_dcgan import Generator
from src.discriminator.d_dcgan import Discriminator
from src.loss.anogan_loss import AnoGANLoss
from src.config import device, latent_dim, lr, beta1, beta2, epochs, lambda_fm

class AnoGANModel:
    def __init__(self, img_channels=1):
        self.G = Generator(latent_dim, img_channels).to(device)
        self.D = Discriminator(img_channels).to(device)
        self.loss_fn = AnoGANLoss(lambda_fm=lambda_fm)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

    def train(self, train_loader):
        self.G.train()
        self.D.train()
        for epoch in range(epochs):
            for imgs in train_loader:
                imgs = imgs.to(device)
                z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)

                x_fake = self.G(z)
                D_real, f_real = self.D(imgs)
                D_fake, f_fake = self.D(x_fake.detach())

                loss_D, loss_G_adv = self.loss_fn.gan_loss(D_real, D_fake)
                self.opt_D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                D_fake, f_fake = self.D(x_fake)
                loss_G = self.loss_fn.total_loss(imgs, x_fake, f_real, f_fake) + loss_G_adv
                self.opt_G.zero_grad()
                loss_G.backward()
                self.opt_G.step()

    def detect_anomaly(self, x_query, steps=500, lr_mapping=0.01):
        self.G.eval()
        self.D.eval()
        z = torch.randn(1, latent_dim, 1, 1, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr_mapping)
        x_query = x_query.to(device)

        for _ in range(steps):
            x_fake = self.G(z)
            _, f_real = self.D(x_query)
            _, f_fake = self.D(x_fake)
            loss = self.loss_fn.total_loss(x_query, x_fake, f_real, f_fake)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_fake = self.G(z)
        _, f_real = self.D(x_query)
        _, f_fake = self.D(x_fake)
        score = self.loss_fn.anomaly_score(x_query, x_fake, f_real, f_fake)
        residual = torch.abs(x_query - x_fake)
        return score, residual
