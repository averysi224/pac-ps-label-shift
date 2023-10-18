import torch as tc

def to_device(x, device):
    if tc.is_tensor(x):
        x = x.to(device)  
    elif isinstance(x, dict):
        x = {k: v.to(device) for k, v in x.items()}
    elif isinstance(x, list):
        x = [to_device(x_i, device) for x_i in x]
    elif isinstance(x, tuple):
        x = (to_device(x_i, device) for x_i in x)
    else:
        raise NotImplementedError
    return x


