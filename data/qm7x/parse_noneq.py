import time
import torch
import os
import math
import h5py
import numpy as np
from ase import Atoms
from morered.datasets import QM7X
from schnetpack.data.loader import _atoms_collate_fn


# Copied from morered.datasets.qm7x
# forces = "forces"  # total ePBE0+MBD forces
energy = "energy"  # ePBE0+MBD: total energy
# Eat = "Eat"  # atomization energy using PBE0 per atom and ePBE0+MBD total energy
# EPBE0 = "EPBE0"  # ePBE0: total energy at the level of PBE0
# EMBD = "EMBD"  # eMBD: total energy at the level of MBD
# FPBE0 = "FMBD"  # FPBE0: total ePBE0 forces
# FMBD = "FMBD"  # FMBD: total eMBD forces
RMSD = "rmsd"  # root mean square deviation from the equilibrium structure

property_unit_dict = {
    # forces: "eV/Ang",
    energy: "eV",
    # Eat: "eV",
    # EPBE0: "eV",
    # EMBD: "eV",
    # FPBE0: "eV/Ang",
    # FMBD: "eV/Ang",
    RMSD: "Ang",
}

property_dataset_keys = {
    # forces: "totFOR",
    energy: "ePBE0+MBD",
    # Eat: "eAT",
    # EPBE0: "ePBE0",
    # EMBD: "eMBD",
    # FPBE0: "pbe0FOR",
    # FMBD: "vdwFOR",
    RMSD: "sRMSD",
}


def load_system_by_key(
    key: tuple[int,int,int],
    noneq_idx: int,
    base_dir: str,
    chunk_size: int = 1000
) -> dict:
    """
    주어진 key=(smi, stereo, conf, step)에 대응하는 시스템을 HDF5에서 읽어 반환합니다.

    Args:
        key: (smiles_id, stereo_iso_id, conform_id, step_id)
        base_dir: .hdf5 파일들이 있는 디렉토리 경로
        chunk_size: 한 파일에 묶여있는 smiles_id 개수 (default=1000)

    Returns:
        {
            "atoms": ase.Atoms, 
            "properties": { prop_name: np.ndarray, … }
        }
    """
    smi, stereo, conf = key
    if noneq_idx is None:
        noneq_key = "opt"
    else:
        noneq_key = f"d{noneq_idx}"


    # 1) 파일 이름 결정: (1~1000)->1000.hdf5, (1001~2000)->2000.hdf5 ...
    file_id = math.ceil(smi / chunk_size) * chunk_size
    fn = os.path.join(base_dir, f"{file_id}.hdf5")

    # 2) 그룹 이름 포맷
    grp_name = f"Geom-m{smi}-i{stereo}-c{conf}-{noneq_key}"

    with h5py.File(fn, "r") as f:
        # 보통 mol_id 별로 그룹이 나눠져 있으면 f[str(smi)][grp_name], 
        # 아니라면 바로 f[grp_name] 에 있을 수 있습니다.
        if str(smi) in f and grp_name in f[str(smi)]:
            conf_ds = f[str(smi)][grp_name]
        else:
            conf_ds = f[grp_name]

        # 3) Atoms 객체 생성
        atoms = Atoms(positions=conf_ds["atXYZ"][()], numbers=conf_ds["atNUM"][()])

        # 4) properties 읽어서 dict으로
        props = {
            key: np.array(conf_ds[property_dataset_keys[key]][()], dtype=np.float64)
            for key in property_unit_dict
        }

    # 3) Tensor 변환
    positions = torch.tensor(atoms.get_positions(),    dtype=torch.float64)  # [N,3]
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)  # [N]

    # properties도 하나씩 Tensor로 변환
    properties = {
        name: torch.tensor(arr, dtype=torch.float64).view(-1)  # e.g. [1]
        for name, arr in props.items()
    }
    return {
        "_positions":      positions,
        "_atomic_numbers": atomic_numbers,
        "_n_atoms": torch.tensor([len(atoms)], dtype=torch.long),
        **properties,
        # "properties":      properties
    }


if __name__ == "__main__":

    # Copied from morered.datasets.qm7x
    energy = "energy"  # ePBE0+MBD: total energy
    EPBE0 = "EPBE0"  # ePBE0: total energy at the level of PBE0
    EMBD = "EMBD"  # eMBD: total energy at the level of MBD
    RMSD = "rmsd"  # root mean square deviation from the equilibrium structure

    property_unit_dict = {
        energy: "eV",
        EPBE0: "eV",
        EMBD: "eV",
        RMSD: "Ang",
    }

    property_dataset_keys = {
        energy: "ePBE0+MBD",
        EPBE0: "ePBE0",
        EMBD: "eMBD",
        RMSD: "sRMSD",
    }

    # Path to the QM7x dataset (change to fit your setup)
    base_dir = "/home/jeheon/programs/MoreRed/src/morered/data"
    datapath = os.path.join(base_dir, "qm7x.db")
    split_file_path = "/home/jeheon/programs/MoreRed/src/morered/configs/data/split.npz"

    timing = False
    if timing:
        st = time.perf_counter()
        data = [load_system_by_key((10, 1, 1), i, base_dir=base_dir) for i in range(1, 100)]
        print(f"Time taken: {time.perf_counter() - st:.2f} seconds")
        print(data)


    data_eq = QM7X(
        datapath=datapath,
        raw_data_path=None,
        remove_duplicates=True,
        only_equilibrium=True,
        batch_size=1,
        split_file=split_file_path,
        num_workers=8,
        distance_unit="Ang",
        property_units={"energy": "eV"},
        load_properties=["energy", "rmsd"],
    )
    data_eq.setup()
    metadata = data_eq.dataset.metadata

    batch = next(iter(data_eq.test_dataloader()))
    smiles_ids = [metadata["groups_ids"]["smiles_id"][i] for i in batch["_idx"]]
    stereo_iso_ids = [
        metadata["groups_ids"]["stereo_iso_id"][i] for i in batch["_idx"]
    ]
    conform_ids = [
        metadata["groups_ids"]["conform_id"][i] for i in batch["_idx"]
    ]
    print("batch=\n", batch)


    data_list = []
    for sid, iid, cid in zip(smiles_ids, stereo_iso_ids, conform_ids):
        key = (sid, iid, cid)
        print("key = ", key)
        data = load_system_by_key(key, None, base_dir=base_dir)
        data_list.append(data)
    print("_atoms_collate_fn(data_list)=\n", _atoms_collate_fn(data_list))
