import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from ase.io import read

import sys
sys.path.append("/home/share/DATA/NeuralOpt/Interpolations/Geodesic_interp")
from get_geodesic_energy import morse_scaler, ATOMIC_RADIUS


def cost_function(wa_c, f, rab, wab):
    """
    args:
        wa_c <np.ndarray>: initial guess position
        f <float>: interpolating ratio
        rab <function>: return target distance
        wab <function>: return contraint positions

    """
    wa_c = wa_c.reshape(-1, 3)
    rab_c = pdist(wa_c)

    rab_i = rab(f)  # target distance
    wa_i = wab(f)  # LST constraint

    first_term = np.sum(
        ((rab_i - rab_c)**2) / (rab_i**4)
    )
    second_term = 1e-6 * np.sum(
        (wa_i - wa_c)**2
    )
    return first_term + second_term


def get_re(atoms, threshold=np.inf):
    from scipy.spatial import KDTree

    rijset = set()
    tree = KDTree(atoms.positions)
    pairs = tree.query_pairs(threshold)
    rijset.update(pairs)
    rijlist = sorted(rijset)

    radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms.get_chemical_symbols()])
    re = np.array([radius[i] + radius[j] for i, j in rijlist])
    return re


def least_square(guess_atoms, target_d, q_type="distance"):
    assert q_type in ["distance", "morse"]
    if q_type == "distance":
        scaler = pdist
    elif q_type == "morse":
        re = get_re(guess_atoms)
        alpha, beta = 1.7, 0.01
        q_ij = lambda x: morse_scaler(re, alpha, beta)(x)[0]
        scaler = lambda x: q_ij(pdist(x))
    else:
        raise NotImplementedError

    guess_pos = guess_atoms.positions

    gtol = 1e-4
    minimize_kwargs = {
        "method": "L-BFGS-B",
        "options": {
            "gtol": gtol,
        }
    }

    def cost_function(trial_pos, target_q):
        wa_c = trial_pos.reshape(-1, 3)
        rab_c = scaler(wa_c)

        if q_type == "distance":
            loss = np.sum(
                ((target_q - rab_c)**2) / (target_q**4)
            )
        elif q_type == "morse":
            # loss = (target_q - rab_c).norm()
            loss = np.linalg.norm(target_q - rab_c)
        else:
            raise NotImplementedError
        print(f"loss = {loss}")
        return loss

    res = minimize(
        cost_function,
        x0=guess_pos.reshape(-1),
        args=(target_d,),
        **minimize_kwargs,
    )
    print(f"success: {res.success}")
    return res.x.reshape(-1, 3)


def interpolate_LST(init_pos, final_pos, p=0.5):
    """Return interpolated molecular geometry from two end-points

    args:
        init_pos <np.ndarray> : XYZ filename of forward endpoint
        final_pos <np.ndarray> : XYZ filename of backward endpoint
        p <float> : interpolating ratio
    return:
        atoms <np.ndarray> : atoms object with interpolated positions
    """
    # atoms = read(init_xyz).copy()
    # coords3d = np.array((read(init_xyz).positions, read(final_xyz).positions))
    coords3d = np.array((init_pos, final_pos))
    # Calculate the condensed distances matrices
    pdists = [pdist(c3d) for c3d in coords3d]


    def rab_(f, pdist_r, pdist_p):
        """Difference in internuclear distances."""
        return (1-f)*pdist_r + f*pdist_p
    rab = lambda f: rab_(f, pdists[0], pdists[1])

    def wab_(f, coords_r, coords_p):
        """Difference in actual cartesian coordinates."""
        return (1-f)*coords_r + f*coords_p
    wab = lambda f: wab_(f, coords3d[0], coords3d[1])

    gtol = 1e-4
    minimize_kwargs = {
        "method": "L-BFGS-B",
        "options": {
            "gtol": gtol,
        }
    }

    x0_flat = wab(p)
    res = minimize(cost_function, x0=x0_flat, args=(p, rab, wab),
                    **minimize_kwargs)
    # print(f"p={p:.04f}, success: {res.success}")
    return res.x.reshape(-1, 3)
    # atoms.set_positions(res.x.reshape(-1, 3))
    # return atoms


if __name__ == "__main__":
    from ase.units import Bohr, Ang
    Bohr2Ang = Bohr / Ang

    init_xyz = "./forward_end_opt.xyz"
    final_xyz = "./backward_end_opt.xyz"
    init_xyz = "/home/share/DATA/jhwoo_TS/gfn2_pysis_try1/wb97xd3-rxn000004-ts000004/IRC/ref_R.xyz"
    final_xyz = "/home/share/DATA/jhwoo_TS/gfn2_pysis_try1/wb97xd3-rxn000004-ts000004/IRC/ref_P.xyz"

    # lst_geom2 = interpolate_LST(init_xyz, final_xyz, p=0.5)
    # print(f"LST geom: \n{lst_geom2.positions}")
    init_pos = read(init_xyz).positions
    final_pos = read(final_xyz).positions

    interp_list = []
    for p in range(1, 10):
        tmp = interpolate_LST(init_pos, final_pos, p / 10)
        interp_list.append(tmp)

    # write trj file
    from ase.io import write
    # filename = "lst_geom.xyz"
    filename = "lst_geom_test.xyz"
    atoms_tmp = read(init_xyz).copy()
    read(init_xyz).write(filename, append=False)
    for interp in interp_list:
        atoms = atoms_tmp.copy()
        atoms.set_positions(interp)
        atoms.write(filename, append=True)
    read(final_xyz).write(filename, append=True)
    print(f"write {filename}")
