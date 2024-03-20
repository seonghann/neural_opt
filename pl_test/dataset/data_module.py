import pathlib
import os

from omegaconf import OmegaConf
import random

from ase import io
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
import torch

from dataset.process_smarts import process_smarts


class AbstractDataModule(LightningDataset):
    def __init__(self, config, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=config.train.batch_size,  # if 'debug' not in config.general.name else 2,
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

    def __init__(
            self,
            root,
            raw_datadir,
            transform=None,
            pre_transform=None,
            stage=None,
            dtype=torch.float32,
            seed=42,
        ):
        self.stage = stage
        assert stage in ["train", "valid", "test"]
        self.file_idx = ["train", "valid", "test"].index(stage)
        self.dtype = dtype
        self.seed = seed
        self.raw_datadir = raw_datadir
        super(GrambowDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_root(self):
        # return '/home/share/DATA/NeuralOpt/SQM_data/'
        return './'

    @property
    def processed_file_names(self):
        return ['train_proc.pt', 'valid_proc.pt', 'test_proc.pt']

    def split_index(self, num_data):
        random.seed(self.seed)
        total_index = list(range(num_data))
        random.shuffle(total_index)

        if self.file_idx == 0:  # train, 80%
            # index = total_index[:int(num_data * 0.8)]
            index = total_index[:1]
        elif self.file_idx == 1:
            index = total_index[int(num_data * 0.8):int(num_data * 0.9)]
        else:
            index = total_index[int(num_data * 0.9):]
        return index

    def process(self):
        import glob

        data_path = os.path.join(self.raw_root, self.raw_datadir, "*")
        print(f"Info] (GrambowDataset) \n\tdata_path: {data_path}")
        # read position and atomic number
        xyz_list = glob.glob(data_path)
        print(f"Info] The number of data : {len(xyz_list)}")

        index = self.split_index(len(xyz_list))
        # make torch geometric data
        data_list = []
        for i, xyz_file in enumerate(xyz_list):
            if i not in index:
                continue

            atoms_list = list(io.iread(xyz_file))
            atoms = atoms_list[0]

            # extract info
            rxn_idx = atoms.info["idx"]
            rxn_smarts = atoms.info["rxn_smarts"]
            # process smarts, extract 2D based information
            atom_type, edge_index, r_edge_type, p_edge_type, r_feat, p_feat = process_smarts(rxn_smarts)

            # make it directed edge
            i, j = edge_index
            mask = i < j
            edge_index = edge_index[:, mask]
            r_edge_type = r_edge_type[mask]
            p_edge_type = p_edge_type[mask]

            # sort edge_index
            i, j = edge_index
            E = edge_index.size(1)
            sort_key = i * E + j
            edge_index = edge_index[:, sort_key.argsort()]
            r_edge_type = r_edge_type[sort_key.argsort()]
            p_edge_type = p_edge_type[sort_key.argsort()]

            pos = [
                torch.tensor(atoms.get_positions(), dtype=self.dtype)
                for atoms in atoms_list
            ]
            pos = torch.stack(pos, dim=0).transpose(0, 1)  # pos shape : (N, T, 3)

            geodesic_length = [
                torch.tensor(atoms.info["GeodesicLength"], dtype=self.dtype)
                for atoms in atoms_list
            ]
            geodesic_length = torch.tensor(geodesic_length).unsqueeze(0)  # geodesic_length shape : (1, T)

            data = Data(
                x=atom_type,
                pos=pos,
                edge_index=edge_index,
                edge_feat_r=r_edge_type,
                edge_feat_p=p_edge_type,
                r_feat=r_feat,
                p_feat=p_feat,
                rxn_idx=rxn_idx,
                rxn_smarts=rxn_smarts,
                geodesic_length=geodesic_length,
            )
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
        print(f"Saved data to {self.processed_paths[self.file_idx]}")


class GrambowDataModule(AbstractDataModule):
    def __init__(self, config):
        self.datadir = config.dataset.datadir
        self.raw_datadir = config.dataset.raw_datadir

        # base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        print(f"debug] (GrambowDataModule) \n\tbase_path: {base_path}", end="")
        root_path = os.path.join(base_path, self.datadir)
        print(f"\n\troot_path: {root_path}")

        datasets = {
            "train": GrambowDataset(root=root_path, raw_datadir=self.raw_datadir, stage="train"),
            "val": GrambowDataset(root=root_path, raw_datadir=self.raw_datadir, stage="valid"),
            "test": GrambowDataset(root=root_path, raw_datadir=self.raw_datadir, stage="test")
        }
        super().__init__(config, datasets)


if __name__ == "__main__":

    # from torch_geometric.loader import DataLoader
    # read config.yaml file
    config = OmegaConf.load("../configs/config.yaml")
    print(config)
    datamodule = GrambowDataModule(config)
