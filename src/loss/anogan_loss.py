import torch
import torch.nn as nn

class AnoGANLoss:
    def __init__(self, lambda_fm=0.5):
        self.lambda_fm = lambda_fm
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

    def gan_loss(self, D_real, D_fake):
        loss_D = -torch.mean(torch.log(D_real + 1e-8) + torch.log(1 - D_fake + 1e-8))
        loss_G = -torch.mean(torch.log(D_fake + 1e-8))
        return loss_D, loss_G

    def residual_loss(self, x_real, x_fake):
        return self.l1_loss(x_real, x_fake)

    def discrimination_loss(self, f_real, f_fake):
        return self.l1_loss(f_real, f_fake)

    def total_loss(self, x_real, x_fake, f_real, f_fake):
        Lr = self.residual_loss(x_real, x_fake)
        Ld = self.discrimination_loss(f_real, f_fake)
        return (1 - self.lambda_fm) * Lr + self.lambda_fm * Ld

    def anomaly_score(self, x_real, x_fake, f_real, f_fake):
        return self.total_loss(x_real, x_fake, f_real, f_fake).item()
