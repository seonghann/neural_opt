import numpy as np
from scipy.spatial.distance import pdist
from lst_interp import least_square

from ase.units import Bohr, Ang
Bohr2Ang = Bohr / Ang


init_xyz = "./forward_end_opt.xyz"
final_xyz = "./backward_end_opt.xyz"
init_xyz = "/home/share/DATA/jhwoo_TS/gfn2_pysis_try1/wb97xd3-rxn000004-ts000004/IRC/ref_R.xyz"
final_xyz = "/home/share/DATA/jhwoo_TS/gfn2_pysis_try1/wb97xd3-rxn000004-ts000004/IRC/ref_P.xyz"





from ase.io import read, iread, write

atoms_0 = list(iread("wb97xd3/wb97xd3_rxn_ts.xyz"))
atoms_T = list(iread("pm7/pm7_rxn_ts.xyz"))


idx = [42,60,78,238,304,344,458,718,746,774,884,913][9]
idx = int(input("idx: "))
print(f"idx: {idx}")

x_T = atoms_T[idx]
file_name = list(x_T.info.keys())[0].split("/")[-1].split(".")[0]
gt_idx = int(file_name[2:])
x_0 = atoms_0[gt_idx]


q_type = ["distance", "morse"][1]
if q_type == "distance":
    d_0 = pdist(x_0.positions)
    x_pred = least_square(x_T, d_0, q_type="distance")
else:
    def make_qij(atoms):
        from lst_interp import get_re
        from get_geodesic_energy import morse_scaler, ATOMIC_RADIUS

        re = get_re(atoms)
        alpha, beta = 1.7, 0.01
        q_ij = lambda x: morse_scaler(re, alpha, beta)(x)[0]
        scaler = lambda x: q_ij(pdist(x))
        # return scaler(atoms.positions)
        return scaler

    qij = make_qij(x_0)

    d_0 = qij(x_0.positions)
    x_pred = least_square(x_T, d_0, q_type="morse")
    print(f"q norm: np.linalg.norm(qij(x_pred) - qij(x_0))={np.linalg.norm(qij(x_pred) - qij(x_0.positions))}")

d_pred = pdist(x_pred)


# print(f"Debug: d_pred - d_0 = \n{d_pred - d_0}")
d_0 = pdist(x_0.positions)
print(f"DMAE: abs(d_pred - d_0).mean() = {abs(d_pred - d_0).mean()}")



tmp = x_0.copy()
tmp.set_positions(x_pred) 
print(f"idx: {idx}")
print(f"gt_idx: {gt_idx}")
write("x_0.xyz", x_0); print("write x_0.xyz")
write("x_pred.xyz", tmp); print("write x_pred.xyz")
write("x_T.xyz", x_T); print("write x_T.xyz")
