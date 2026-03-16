import torch.nn as nn

def coerce_module_to_device(module: nn.Module, device):
    """
    Ensures all submodules and buffers are on the target device.
    Useful for hybrid SNN/ANN modules that might initialize buffers lazily.
    """
    module.to(device)
    for child in module.children():
        child.to(device)
