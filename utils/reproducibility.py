import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """
    Mengunci status stokastik untuk reproduktibilitas penuh di tingkat Python, Numpy, dan CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN Deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # OS Hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
