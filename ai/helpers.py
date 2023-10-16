import torch


def get_device(device="cuda"):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    use_mps = device == "mps" and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device(device)
    elif use_mps:
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    return device
