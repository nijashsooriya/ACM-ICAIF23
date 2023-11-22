import os
import copy
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from unet import UNet
import modules
import pickle
import argparse

class Diffusion:
    def __init__(self, timesteps = 1000, beta_start = 1e-4, beta_end = 0.02, data_size = 1000,device = "cpu"):
        self.steps =  timesteps
        self.start = beta_start
        self.end = beta_end
        self.beta = self.noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim= 0)
        self.size = data_size
        self.device = device

    #Noise scheduling
    def noise_schedule(self):
        return torch.linspace(self.start, self.end, self.steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.steps, size=(n,))

    def noise_data(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def mean(self,x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return ((1/sqrt_alpha_hat)*(x - (self.beta/sqrt_one_minus_alpha_hat)*epsilon))

    def sample(self, model, n):
        #Evaluation, switch layers
        model.eval()
        with torch.no_grad():
            #Creates a tensor of 4 dimensions with specified parameter lengths
            x = torch.randn((n, 3, self.data_size, self.data_size)).to(self.device)
            for i in reversed(range(1, self.steps), position = 0):
                t = (torch.ones(n)*i).long().to(self.device)
                pred_epsilon =- model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    epsilon = torch.randn_like(x)
                else:
                    epsilon = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_epsilon) + torch.sqrt(beta) * epsilon
            model.train()
            return x


def train(args):
    device = args.device
    model = UNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    with open('val_data.pkl', 'rb') as f:
        load_data = pickle.load(f)
    #Create a diffusion object
    diffusion = Diffusion(data_size = args.data_size, device = device)

    for epoch in range(args.epochs):
        for i, (dataset, _) in enumerate(tqdm(load_data)):
            dataset = dataset.to(device)
            t = Diffusion.sample_timesteps(load_data.shape[0]).to(device)
            x_t, epsilon = Diffusion.noise_data(dataset, t)
            predicted_epsilon = model(x_t, t)
            loss = mse(epsilon, predicted_epsilon)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    sampled_data = Diffusion.sample(model, n=load_data.shape[0])
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_TimeSeries"
    args.epochs = 500
    args.batch_size =  10
    args.data_size = 20000
    args.device = "cpu"
    args.data_set_path = r"C:\Users\Nijash Sooriya\Desktop\DiffusionModel\val_data.pkl"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
     launch()