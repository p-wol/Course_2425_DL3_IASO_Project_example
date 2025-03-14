from time import gmtime, strftime
import argparse
import os
import pickle
import json
import numpy as np
import torch
from torch import optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from vae import VAE_FC, train_model_vae

# Arguments
parser = argparse.ArgumentParser(
        prog = "24/25 DL3 IASO Example",
        description = "Example program for the Applciations of DL course (2024/2025 DL3 IASO)")

parser.add_argument("--dataset_path", type = str, default = "", help = "path to the dataset file")
parser.add_argument("--batch_size", type = int, default = 128, help = "batch size")
parser.add_argument("--save_dir", type = str, default = "save", help = "where to save the model, the logs and the configuration")
parser.add_argument("--lambda_reconstruct", type = float, default = .5, help = "VAE loss: renconstruction term")
parser.add_argument("--lambda_kl", type = float, default = .5, help = "VAE loss: KL term")
parser.add_argument("--latent_dim", type = int, default = 10, help = "VAE model: latent dimension")
parser.add_argument("--nepochs", type = int, default = 10, help = "optimization: number of training epochs")

args = parser.parse_args()

layers = [28**2, 500, 200, 50]

with_cuda = torch.cuda.is_available()
if with_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Create the directory containing the model, the logs, etc.
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
out_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(out_dir)

path_model = os.path.join(out_dir, "model.pkl")
path_config = os.path.join(out_dir, "config.json")
path_logs = os.path.join(out_dir, "logs.json")

# Store the configuration
with open(path_config, "w") as f:
    json.dump(vars(args), f)

# build transform
transform = transforms.Compose([
    transforms.ToTensor(),
    ]) 

# choose the training and test datasets
train_data = datasets.MNIST(args.dataset_path, train = True,
        download = False, transform = transform)
test_data = datasets.MNIST(args.dataset_path, train = False,
        download = False, transform = transform)

train_size = len(train_data)
test_size = len(test_data)

# build the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

# specify the image classes
classes = [f"{i}" for i in range(10)]

# Building the loss
def build_loss_vae(lambda_reconstruct = .5, lambda_kl = .5):
    def loss_vae(x, x_hat, mean, logvar):
        reconstruct_loss = lambda_reconstruct * (x - x_hat).pow(2).sum()
        KL_loss = -lambda_kl * torch.sum(logvar - mean.pow(2) - logvar.exp())

        return reconstruct_loss + KL_loss
    return loss_vae

# Training
model = VAE_FC(layers, latent_dim = args.latent_dim).to(device)
criterion = build_loss_vae(lambda_reconstruct = args.lambda_reconstruct, lambda_kl = args.lambda_kl)
optimizer = optim.Adam(model.parameters())

train_losses, train_acc = train_model_vae(train_loader, model, criterion, optimizer, args.nepochs, device)

torch.save({"vae_fc": model.state_dict()}, path_model)
torch.save({"train_losses": train_losses, "train_acc": train_acc}, path_logs)
#torch.save({"vae_fc": model.state_dict()}, "VAE_FC_FashionMNIST.pkl")

#dct_load = torch.load("VAE_FC_MNIST.pkl", weights_only = True)
#dct_load = torch.load("VAE_FC_FashionMNIST.pkl", weights_only = True)
#model.load_state_dict(dct_load["vae_fc"])
