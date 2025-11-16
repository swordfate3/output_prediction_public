import torch
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nvcc_version = torch.version.cuda
print(device)
print(nvcc_version)
