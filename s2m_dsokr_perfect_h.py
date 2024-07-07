import argparse
import pickle
import os

import numpy as np
from grakel.kernels import VertexHistogram, NeighborhoodSubgraphPairwiseDistance
from grakel.kernels import WeisfeilerLehman, CoreFramework
import matplotlib.pyplot as plt

from Methods.SketchGraphs import SubSampleGraphs
from Utils.s2m_utils import from_grkl_to_dict, to_networkx, eval_graph
from Utils.load_data import load_smi2mol


def compute_sketched_feature_maps(Y, Y_tr, output_kernel, Ry, Ay):
    KRy = Ry.multiply_Gram_one_side(Y_tr, output_kernel, Y=Y)
    Z = (KRy).dot(Ay)
    return Z


def get_graph_kernel(kernel_name, n_jobs):
    if kernel_name == "NSPD":
        return NeighborhoodSubgraphPairwiseDistance(n_jobs=n_jobs, normalize=True)
    elif kernel_name == "WL-VH":
        return WeisfeilerLehman(
            n_jobs=n_jobs, normalize=True, base_graph_kernel=VertexHistogram
        )
    elif kernel_name == "CORE-WL":
        return CoreFramework(
            n_jobs=n_jobs, normalize=True, base_graph_kernel=WeisfeilerLehman
        )
    else:
        return None


def get_score(sketch, my, Y_gkl_tr, Y_gkl_val, Y_dict_val, n_jobs):
    if sketch == "NonSketch":
        output_kernel = get_graph_kernel(kernel_name, n_jobs)
        Y_gkl_cand = Y_gkl_tr
        output_kernel.fit(Y_gkl_cand)
        scores_val = output_kernel.transform(Y_gkl_val)
        idx_pred_val = np.argmax(scores_val, axis=1)
        Y_gkl_pred_val = Y_gkl_tr[idx_pred_val]
        Y_dict_pred_val = from_grkl_to_dict(Y_gkl_pred_val)
        G_preds_val = [
            to_networkx(y_dict_pred) for y_dict_pred in Y_dict_pred_val
        ]
        G_trgts_val = [to_networkx(y_dict_val) for y_dict_val in Y_dict_val]
        mean_eds_val = eval_graph(G_preds_val, G_trgts_val, n_jobs=n_jobs)[
            "edit_distance"
        ]
    else:
        print(my)
        n_tr = len(Y_gkl_tr)
        output_kernel = get_graph_kernel(kernel_name, n_jobs)
        if sketch == "SubSample":
            Ry = SubSampleGraphs((my, n_tr))
        else:
            raise ValueError
        KRy = Ry.multiply_Gram_one_side(Y_gkl_tr, output_kernel, Y=Y_gkl_tr)
        RyKRyT = Ry.multiply_matrix_one_side(KRy, right=False)
        V, D, _ = np.linalg.svd(RyKRyT)
        nnz_D = np.logical_not(
            np.isclose(
                D,
                np.zeros(D.shape),
                rtol=1e-12,
            )
        )
        D_r, V_r = D[nnz_D], V[:, nnz_D]
        Ay = (D_r ** (-1 / 2)) * V_r

        # ry = Ay.shape[1]

        Z_tr = compute_sketched_feature_maps(
            Y_gkl_tr, Y_gkl_tr, output_kernel, Ry, Ay
        )
        Z_val = compute_sketched_feature_maps(
            Y_gkl_val, Y_gkl_tr, output_kernel, Ry, Ay
        )
        Z_c = Z_tr.copy()

        scores_val = Z_val.dot(Z_c.T)
        idx_pred_val = np.argmax(scores_val, axis=1)
        Y_gkl_pred_val = Y_gkl_tr[idx_pred_val]
        Y_dict_pred_val = from_grkl_to_dict(Y_gkl_pred_val)
        G_preds_val = [
            to_networkx(y_dict_pred) for y_dict_pred in Y_dict_pred_val
        ]
        G_trgts_val = [to_networkx(y_dict_val) for y_dict_val in Y_dict_val]
        mean_eds_val = eval_graph(G_preds_val, G_trgts_val, n_jobs=n_jobs)[
            "edit_distance"
        ]

    return mean_eds_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed_split", type=int, default=64)

    args = parser.parse_args()
    # Setting random seed
    np.random.seed(seed=42)

    (
        X_smiles_tr,
        Y_gkl_tr,
        Y_dict_tr,
        X_smiles_val,
        Y_gkl_val,
        Y_dict_val,
        X_smiles_te,
        Y_gkl_te,
        Y_dict_te,
    ) = load_smi2mol(
        path="Data/smi2mol/",
        n_valid=500,
        n_test=2000,
        delete_atoms=True,
        random_seed=args.random_seed_split,
    )

    mys = [50, 100, 200, 400, 800, 1600, 3200, 6400]

    save_dir = f"exper/s2m_seed{args.random_seed_split}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_tr = len(X_smiles_tr)

    sketchers = ["SubSample", "NonSketch"]
    colormaps = {
        "SubSample": "tab:blue",
        "NonSketch": "tab:olive",
    }

    n_jobs = 16
    output_kernels = ["WL-VH", "CORE-WL", "NSPD"]

    i = 0
    for kernel_name in output_kernels:
        fig, axes = plt.subplots(
            1, 1, figsize=(4, 4)
        )

        all_eds = {}
        for sketch in sketchers:
            all_eds[sketch] = []
            if sketch == "NonSketch":
                mean_eds_val = get_score(
                    sketch, None, Y_gkl_tr, Y_gkl_val, Y_dict_val, n_jobs)
                all_eds[sketch] = [mean_eds_val] * len(mys)
            else:
                for my in mys:
                    mean_eds_val = get_score(
                        sketch, my, Y_gkl_tr, Y_gkl_val, Y_dict_val, n_jobs)
                    all_eds[sketch].append(mean_eds_val)
        for key, item in all_eds.items():
            axes.plot(mys, item, ".-", label=key, color=colormaps[key])
        axes.set_ylabel("GED")
        axes.set_ylim(0.0)
        axes.set_xlabel(r"$m_y$")
        axes.set_title(kernel_name)
        axes.legend()
        i += 1
        with open(os.path.join(save_dir, f"h_perfect_{kernel_name}.pkl"), 'wb') as f:
            pickle.dump(all_eds, f)
        fig.savefig(os.path.join(save_dir, f"h_perfect_{kernel_name}.pdf"))
