import pathlib
import os

from omegaconf import OmegaConf
import random

from ase import io
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.utils import subgraph
import torch_geometric as torch_g
import torch


class AbstractDataModule(LightningDataset):
    def __init__(self, config, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=config.train.batch_size if 'debug' not in config.general.name else 2,
            num_workers=config.train.num_workers,
            pin_memory=getattr(config.dataset, 'pin_memory', False),
        )
        self.config = config
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_coutns(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_datalader(), self.val_dataloader()]:
            for data in loader:
                uniuqe, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts


class GrambowDataset(InMemoryDataset):
    r"""The Grambow dataset from the `"Grambow et al. (2020)" <https://arxiv.org/abs/2006.00897>`_ paper,
    containing 1000 reaction transition states.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, root, transform=None, pre_transform=None, stage=None):
        self.stage = stage
        assert stage in ["train", "valid", "test"]
        self.file_idx = ["train", "valid", "test"].index(stage)
        super(GrambowDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_dir(self):
        return '/home/ksh/MolDiff/NeuralOpt/neural_opt/data'

    @property
    def raw_file_name(self):
        return ['wb97xd3/wb97xd3_rxn_ts.xyz', 'wb97xd3_geodesic/wb97xd3_geodesic_rxn_ts.xyz']

    @property
    def processed_file_names(self):
        return ['train_proc.pt', 'valid_proc.pt', 'test_proc.pt']

    def split_index(self, num_data, seed=42):
        random.seed(seed)
        total_index = list(range(num_data))
        random.shuffle(total_index)

        if self.file_idx == 0:  # train, 80%
            index = total_index[:int(num_data * 0.8)]
        elif self.file_idx == 1:
            index = total_index[int(num_data * 0.8):int(num_data * 0.9)]
        else:
            index = total_index[int(num_data * 0.9):]
        return index

    def process(self):
        data_list = []
        data_path1 = os.path.join(self.raw_file_dir, self.raw_file_name[0])
        data_path2 = os.path.join(self.raw_file_dir, self.raw_file_name[1])
        print(f"Info] (GrambowDataset) \n\tdata_path1: {data_path1}\n\tdata_path2: {data_path2}")
        atom_list1 = list(io.iread(data_path1, index=":"))
        atom_list2 = list(io.iread(data_path2, index=":"))

        print(atom_list1[0], len(atom_list1))
        print(atom_list2[0], len(atom_list2))
        assert len(atom_list1) == len(atom_list2)
        # read position and atomic number
        pos_list1 = [atom.positions for atom in atom_list1]
        pos_list2 = [atom.positions for atom in atom_list2]
        atom_list1 = [atom.numbers for atom in atom_list1]
        atom_list2 = [atom.numbers for atom in atom_list2]

        index = self.split_index(len(atom_list1))
        # make torch geometric data
        for i, (pos1, pos2, atom1, atom2) in enumerate(zip(pos_list1, pos_list2, atom_list1, atom_list2)):
            if i not in index:
                continue

            pos1 = torch.tensor(pos1, dtype=torch.float64)
            pos2 = torch.tensor(pos2, dtype=torch.float64)
            # pos1 = torch.tensor(pos1, dtype=torch.float)
            # pos2 = torch.tensor(pos2, dtype=torch.float)
            atom1 = torch.tensor(atom1, dtype=torch.long)
            atom2 = torch.tensor(atom2, dtype=torch.long)
            edge_index = torch_g.nn.radius_graph(pos1, r=20.0, batch=None, loop=False)
            # make it directed edge
            i, j = edge_index
            mask = i < j
            edge_index = edge_index[:, mask]

            # sort edge_index
            i, j = edge_index
            E = edge_index.size(1)
            sort_key = i * E + j
            edge_index = edge_index[:, sort_key.argsort()]
            data = Data(x=atom1, pos_1=pos1, pos_2=pos2, edge_index=edge_index)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
        print(f"Saved data to {self.processed_paths[self.file_idx]}")


class GrambowDataModule(AbstractDataModule):
    def __init__(self, config):
        self.datadir = config.dataset.datadir

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        print(f"debug] (GrambowDataModule) \n\tbase_path: {base_path}", end="")
        root_path = os.path.join(base_path, self.datadir)
        print(f"\n\troot_path: {root_path}")

        datasets = {
            "train": GrambowDataset(root=root_path, stage="train"),
            "val": GrambowDataset(root=root_path, stage="valid"),
            "test": GrambowDataset(root=root_path, stage="test")
        }
        super().__init__(config, datasets)


if __name__ == "__main__":

    # read config.yaml file
    config = OmegaConf.load("config.yaml")
    print(config)
    datamodule = GrambowDataModule(config)
