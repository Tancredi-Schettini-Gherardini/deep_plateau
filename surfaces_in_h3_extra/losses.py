import torch

def L2_squared(t):
    return (t**2).sum(dim=-1).mean()

def L2(t):
    return ((t**2).sum(dim=-1).mean())**(1/2)

def Lp(t, p=4):
    return ((torch.abs(t)**p).sum(dim=-1).mean())**(1/p)

def Lp_pow(t, p=4):
    return (torch.abs(t)**p).sum(dim=-1).mean()