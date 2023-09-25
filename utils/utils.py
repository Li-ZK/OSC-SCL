import json
import torch
import random
import os
import numpy as np

from .pyExt import Dict2Obj

def getDatasetInfo(dataset):
    with open("datasets/dataset_config.json", "r") as f:
        info = json.load(f)[dataset]

    return Dict2Obj(info)

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
