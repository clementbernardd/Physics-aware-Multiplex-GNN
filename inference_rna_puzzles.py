import os
import os.path as osp
import tqdm
import argparse
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import DataLoader
from typing import Optional

import sys
sys.path.append('pamnet')
from models import PAMNet, Config
from datasets import TUDataset


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def predict(in_path: str, dataset: str, batch_size: int, out_path: Optional[str], model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
        except ValueError:
            torch.cuda.set_device(0)
    set_seed(40)
    test_dataset = TUDataset(in_path, name=dataset, use_node_attr=True)

    # Load dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    y_hat_dict = {}
    name_list = np.loadtxt(osp.join(in_path, dataset, 'raw',
                                    dataset + '_graph_names.txt'), dtype=str,
                           converters={0: lambda s: s[:-4]})
    name_list = [x+".pdb" for x in name_list]
    names_to_avoid = []
    for index, data in enumerate(tqdm.tqdm(test_loader)):
        data = data.to(device)
        if name_list[index] in names_to_avoid:
            y_hat_dict[name_list[index]] = [np.nan]
            continue
        try:
            output = model(data)
        except RuntimeError:
            continue
        c_out = output.reshape(-1).tolist()
        c_out = [out for out in c_out if out > 1e-10]
        y_hat_dict[name_list[index]] = c_out
    df = pd.DataFrame.from_dict(y_hat_dict, orient='index')
    df.columns = ['PAMNet']
    if out_path is not None:
        df.to_csv(out_path, sep=',', index=True)
    return df
