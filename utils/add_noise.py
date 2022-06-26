import copy
import numpy as np


def add_noise(x, noise_type, noise_params):

    if noise_type == "Gaussian":
        G_mean = noise_params[0]
        G_std = noise_params[1] * np.std(x)
        Gaussian_noise = np.random.normal(loc=G_mean, scale=G_std, size=x.shape)
        return x+Gaussian_noise

    if noise_type == "Poisson":
        P_lambda = noise_params[0] * np.std(x)
        Poisson_noise = np.random.poisson(lam=P_lambda, size=x.shape)
        return x+Poisson_noise

    if noise_type == "Salt&Pepper":
        Prob_salt = noise_params[0]
        Prob_pepper = noise_params[1]
        SandP_noise = np.random.rand(*(a for a in x.shape))
        Salt_mask = [SandP_noise>Prob_salt]
        Pepper_mask = [SandP_noise<Prob_pepper]

        SandP_img = copy.deepcopy(x)
        SandP_img[tuple(Salt_mask)] = 1
        SandP_img[tuple(Pepper_mask)] = 0
        return SandP_img

    if noise_type == "Speckle":
        S_mean = noise_params[0]
        S_std = noise_params[1]
        Speckle_factor = np.random.normal(loc=S_mean, scale=S_std, size=x.shape)
        return np.multiply(x, 1+Speckle_factor)

    if noise_type == "Racian":
        snr = noise_params[0]
        sigma = np.amax(x) / snr
        noise_1 = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        noise_2 = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        return np.sqrt((x + noise_1) ** 2 + noise_2 ** 2)

    if noise_type == "Rayleigh":
        snr = noise_params[0]
        sigma = np.amax(x) / snr
        noise_1 = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        noise_2 = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
        return x + np.sqrt(noise_1 ** 2 + noise_2 ** 2)

    if noise_type == "None":
        return x