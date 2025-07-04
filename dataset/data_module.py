import pathlib
import os
import glob

from omegaconf import OmegaConf
import random

from ase import io
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.lightning import LightningDataset
import torch

from dataset.process_smarts import process_smarts, process_smarts_single


def load_datamodule(config):
    if config.dataset.type == "reaction":
        datamodule = GrambowDataModule(config)
    elif config.dataset.type == "molecule":
        datamodule = QM9DataModule(config)
    else:
        raise ValueError()
    return datamodule


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

        self.stage = None
        self.file_idx = None
        return

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
            data_split=None,
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
        self.data_split = data_split
        super(GrambowDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)
        return

    @property
    def processed_file_names(self):
        return ['train_proc.pt', 'valid_proc.pt', 'test_proc.pt']

    def split_index(self, num_data):
        random.seed(self.seed)
        total_index = list(range(num_data))
        random.shuffle(total_index)

        if self.file_idx == 0:  # train, 80%
            index = total_index[:int(num_data * 0.8)]
        elif self.file_idx == 1:
            index = total_index[int(num_data * 0.8):int(num_data * 0.9)]
        else:
            index = total_index[int(num_data * 0.9):]
        return index

    def split_index_from_pkl(self):
        import pickle

        path = self.data_split
        with open(path, "rb") as f:
            split_indices = pickle.load(f)

        print(f"data_split pickle file path={path}")
        if self.file_idx == 0:
            index = split_indices["train_index"]
            print(f"Load train_index from {path}. len(train_index)={len(index)}")
        elif self.file_idx == 1:
            index = split_indices["valid_index"]
            print(f"Load valid_index from {path}. len(valid_index)={len(index)}")
        else:
            index = split_indices["test_index"]
            print(f"Load test_index from {path}. len(test_index)={len(index)}")
        return index

    def process(self):
        data_path = os.path.join(self.raw_datadir, "*.xyz")
        print(f"Info] (GrambowDataset) \n\tdata_path: {data_path}")

        # read position and atomic number
        xyz_list = glob.glob(data_path)
        print(f"Info] The number of data : {len(xyz_list)}")

        if self.data_split is None:
            index = self.split_index(len(xyz_list))
        else:
            index = self.split_index_from_pkl()

        # make torch geometric data
        data_list = []
        for i, xyz_file in enumerate(xyz_list):
            atoms_list = list(io.iread(xyz_file))
            atoms = atoms_list[0]

            # extract info
            # rxn_idx = atoms.info["idx"]
            idx = atoms.info["idx"]
            # if rxn_idx not in index:
            if idx not in index:
                continue
            rxn_smarts = atoms.info["rxn_smarts"]
            if not rxn_smarts:
                # check whether rxn_smarts is empty
                continue

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

            if "GeodesicLength" in atoms.info:
                geodesic_length = [
                    torch.tensor(atoms.info["GeodesicLength"], dtype=self.dtype)
                    for atoms in atoms_list
                ]
                geodesic_length = torch.tensor(geodesic_length).unsqueeze(0)  # geodesic_length shape : (1, T)
            else:
                geodesic_length = None

            if "time_step" in atoms.info:
                time_step = torch.tensor(atoms.info["time_step"])
            else:
                time_step = None

            if "q_target" in atoms.info:
                q_target = torch.from_numpy(atoms.info["q_target"]).to(self.dtype)
            else:
                q_target = None

            data = Data(
                x=atom_type,
                pos=pos,
                edge_index=edge_index,
                edge_feat_r=r_edge_type,
                edge_feat_p=p_edge_type,
                r_feat=r_feat,
                p_feat=p_feat,
                # rxn_idx=rxn_idx,
                idx=idx,
                rxn_smarts=rxn_smarts,
                geodesic_length=geodesic_length,
                time_step=time_step,
                q_target=q_target,
            )
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
        print(f"Saved data to {self.processed_paths[self.file_idx]}")
        return


class GrambowDataModule(AbstractDataModule):
    def __init__(self, config):
        self.datadir = config.dataset.datadir
        self.raw_datadir = config.dataset.raw_datadir
        self.data_split = config.dataset.data_split
        self.dtype = torch.float64 if config.dataset.dtype == "float64" else torch.float32

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        print(f"debug] (GrambowDataModule) \n\tbase_path: {base_path}", end="")
        root_path = os.path.join(base_path, self.datadir)
        print(f"\n\troot_path: {root_path}")

        datasets = {
            "train": GrambowDataset(root=root_path, raw_datadir=self.raw_datadir, data_split=self.data_split, stage="train", dtype=self.dtype),
            "val": GrambowDataset(root=root_path, raw_datadir=self.raw_datadir, data_split=self.data_split, stage="valid", dtype=self.dtype),
            "test": GrambowDataset(root=root_path, raw_datadir=self.raw_datadir, data_split=self.data_split, stage="test", dtype=self.dtype)
        }
        super().__init__(config, datasets)
        return


class QM9Dataset(InMemoryDataset):
    def __init__(
        self,
        root,
        raw_datadir,
        data_split=None,
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
        self.data_split = data_split
        super(QM9Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)
        return

    @property
    def processed_file_names(self):
        return ['train_proc.pt', 'valid_proc.pt', 'test_proc.pt']

    def split_index(self, num_data):
        random.seed(self.seed)
        total_index = list(range(num_data))
        random.shuffle(total_index)

        if self.file_idx == 0:  # train, 80%
            index = total_index[:int(num_data * 0.8)]
        elif self.file_idx == 1:
            index = total_index[int(num_data * 0.8):int(num_data * 0.9)]
        else:
            index = total_index[int(num_data * 0.9):]
        return index

    def split_index_from_pkl(self):
        import pickle

        path = self.data_split
        with open(path, "rb") as f:
            split_indices = pickle.load(f)

        print(f"data_split pickle file path={path}")
        if self.file_idx == 0:
            index = split_indices["train_index"]
            print(f"Load train_index from {path}. len(train_index)={len(index)}")
        elif self.file_idx == 1:
            index = split_indices["valid_index"]
            print(f"Load valid_index from {path}. len(valid_index)={len(index)}")
        else:
            index = split_indices["test_index"]
            print(f"Load test_index from {path}. len(test_index)={len(index)}")
        return index

    def process(self):
        data_path = os.path.join(self.raw_datadir, "*.xyz")
        print(f"Info] (QM9Dataset) \n\tdata_path: {data_path}")

        # read position and atomic number
        xyz_list = glob.glob(data_path)
        print(f"Info] The number of data : {len(xyz_list)}")

        if self.data_split is None:
            index = self.split_index(len(xyz_list))
        else:
            index = self.split_index_from_pkl()

        # make torch geometric data
        data_list = []
        for i, xyz_file in enumerate(xyz_list):
            atoms_list = list(io.iread(xyz_file))
            atoms = atoms_list[0]  # reference atoms

            # extract info
            idx = atoms.info["idx"]
            if idx not in index:
                continue
            smarts = atoms.info["smarts"]
            print(f"Debug: i={i}, smarts=\n{smarts}", flush=True)

            # process smarts, extract 2D based information
            try:
                atom_type, edge_index, edge_type, node_feat = process_smarts_single(smarts)
            except:
                print(f"Warning: process_smarts_single failed for i={i}, smarts={smarts}")
                continue

            # make it directed edge
            i, j = edge_index
            mask = i < j
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

            # sort edge_index
            i, j = edge_index
            E = edge_index.size(1)
            sort_key = i * E + j
            edge_index = edge_index[:, sort_key.argsort()]
            edge_type = edge_type[sort_key.argsort()]

            pos = [
                torch.tensor(atoms.get_positions(), dtype=self.dtype)
                for atoms in atoms_list
            ]
            pos = torch.stack(pos, dim=0).transpose(0, 1)  # pos shape : (N, T, 3)

            if "GeodesicLength" in atoms.info:
                geodesic_length = [
                    torch.tensor(atoms.info["GeodesicLength"], dtype=self.dtype)
                    for atoms in atoms_list
                ]
                geodesic_length = torch.tensor(geodesic_length).unsqueeze(0)  # geodesic_length shape : (1, T)
            else:
                geodesic_length = None

            if "time_step" in atoms.info:
                time_step = torch.tensor(atoms.info["time_step"])
            else:
                time_step = None

            if "q_target" in atoms.info:
                q_target = torch.from_numpy(atoms.info["q_target"]).to(self.dtype)
            else:
                q_target = None

            data = Data(
                x=atom_type,
                pos=pos,
                edge_index=edge_index,
                edge_feat=edge_type,
                node_feat=node_feat,
                idx=idx,
                smarts=smarts,
                geodesic_length=geodesic_length,
                time_step=time_step,
                q_target=q_target,
            )
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
        print(f"Saved data to {self.processed_paths[self.file_idx]}")
        return


class QM9DataModule(AbstractDataModule):
    def __init__(self, config):
        self.datadir = config.dataset.datadir
        self.raw_datadir = config.dataset.raw_datadir
        self.data_split = config.dataset.data_split
        self.dtype = torch.float64 if config.dataset.dtype == "float64" else torch.float32

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        print(f"debug] (QM9DataModule) \n\tbase_path: {base_path}", end="")
        root_path = os.path.join(base_path, self.datadir)
        print(f"\n\troot_path: {root_path}")

        datasets = {
            "train": QM9Dataset(root=root_path, raw_datadir=self.raw_datadir, data_split=self.data_split, stage="train", dtype=self.dtype),
            "val": QM9Dataset(root=root_path, raw_datadir=self.raw_datadir, data_split=self.data_split, stage="valid", dtype=self.dtype),
            "test": QM9Dataset(root=root_path, raw_datadir=self.raw_datadir, data_split=self.data_split, stage="test", dtype=self.dtype)
        }
        super().__init__(config, datasets)
        return


if __name__ == "__main__":

    # from torch_geometric.loader import DataLoader
    # read config.yaml file
    config = OmegaConf.load("../configs/config.yaml")
    print(config)
    datamodule = GrambowDataModule(config)
