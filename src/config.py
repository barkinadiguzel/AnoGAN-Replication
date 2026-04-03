import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100

lr = 0.0002
beta1 = 0.5
beta2 = 0.999
batch_size = 64
epochs = 200

lambda_fm = 0.5

img_size = 64
channels = 1 
