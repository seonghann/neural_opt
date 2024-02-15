import pathlib
import os

from omegaconf import OmegaConf
import random

from ase import io
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.data.lightning import LightningDataset
import torch

from dataset.process_smarts import process_smarts


# from torch_geometric.loader import DataLoader
# read config.yaml file
config = OmegaConf.load("../configs/config.yaml")
print(config)
datamodule = GrambowDataModule(config)
