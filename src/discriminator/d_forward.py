def d_forward(discriminator, x):
    out, features = discriminator(x)
    return out, features
