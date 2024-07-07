import argparse
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

from Utils.load_data import load_text2mols
from Utils.chebi_20_utils import gaussian_kernel
from Methods.Sketch import SubSample, pSparsified, Gaussian


def compute_sketched_feature_maps(Y, Y_tr, output_kernel, Ry, Ay):

    KRy = Ry.multiply_Gram_one_side(Y, output_kernel, Y=Y_tr)
    Z = (KRy).dot(Ay)
    return Z


def eval_rank(scores, offset):
    cid_locs = np.argsort(scores, axis=1)[:, ::-1]
    ranks_tmps = np.argsort(cid_locs)
    ranks = []
    for i, item in enumerate(ranks_tmps):
        ranks.append(item[offset + i] + 1)

    ranks = np.array(ranks)

    return {
        "ranks": ranks,
        "mr": np.mean(ranks),
        "mrr": np.mean(1 / ranks),
        "top1": np.mean(ranks <= 1),
        "top10": np.mean(ranks <= 10),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='exper/chebi-20/')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Setting random seed
    np.random.seed(seed=42)
    (
        X_tr,
        Y_dict_tr,
        X_val,
        Y_dict_val,
        X_te,
        Y_dict_te,
        Y_dict_cand,
    ) = load_text2mols(path='Data/chebi-20/')

    Y_tr = np.array([item['mol2vec'] for item in Y_dict_tr])
    Y_val = np.array([item['mol2vec'] for item in Y_dict_val])
    Y_cand = np.array([item['mol2vec'] for item in Y_dict_cand])

    print("Number of candidates: ", len(Y_cand))

    mys = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    n_tr = len(Y_tr)

    sketchers = ['SubSample', 'Gaussian', 'pSparsified', 'NonSketch']
    colormaps = {'SubSample': 'tab:blue',
                 'Gaussian': 'tab:orange',
                 'pSparsified': 'tab:red',
                 'NonSketch': 'tab:cyan'}

    output_kernels = {'cosine': cosine_similarity}
    for gamma in [1e-9, 1e-6, 1e-3, 1]:
        output_kernels[f'gaussian(gamma={gamma})'] = gaussian_kernel(gamma)

    i = 0
    for kernel_name, output_kernel in output_kernels.items():
        fig, axe = plt.subplots(
            1, 1, figsize=(4, 4)
        )
        print(i)
        print(kernel_name)
        all_mrrs = {}
        for sketch in sketchers:
            all_mrrs[sketch] = []
            if sketch == 'NonSketch':
                scores_val = output_kernel(Y_val, Y_cand)
                res_val = eval_rank(scores_val, offset=len(Y_tr))
                all_mrrs[sketch] = [res_val["mrr"]] * len(mys)
            else:
                for my in mys:
                    if sketch == 'SubSample':
                        Ry = SubSample((my, n_tr))
                    elif sketch == 'Gaussian':
                        Ry = Gaussian((my, n_tr))
                    elif sketch == 'pSparsified':
                        p = 20.0 / n_tr
                        Ry = pSparsified((my, n_tr), p=p, type='Rademacher')
                    else:
                        raise ValueError
                    RyKRyT = Ry.multiply_Gram_both_sides(Y_tr, output_kernel)
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

                    ry = Ay.shape[1]

                    Z_tr = compute_sketched_feature_maps(
                        Y_tr, Y_tr, output_kernel, Ry, Ay
                    )
                    Z_val = compute_sketched_feature_maps(
                        Y_val, Y_tr, output_kernel, Ry, Ay
                    )
                    Z_c = compute_sketched_feature_maps(
                        Y_cand, Y_tr, output_kernel, Ry, Ay
                    )

                    scores_val = Z_val.dot(Z_c.T)
                    res_val = eval_rank(scores_val, offset=len(Y_tr))
                    all_mrrs[sketch].append(res_val["mrr"])
        print(all_mrrs)
        with open(os.path.join(args.save_dir, f"hperfet_mol2vec_{kernel_name}.pkl"), 'wb') as f:
            pickle.dump(all_mrrs, f)
        for key, item in all_mrrs.items():
            axe.plot(mys, item, '.-', label=key, color=colormaps[key])
        axe.set_ylabel('MRR')
        axe.set_xlabel(r'$m_y$')
        axe.set_title(kernel_name)
        axe.legend()
        i += 1
        fig.savefig(os.path.join(args.save_dir,
                    f'hperfet_mol2vec_{kernel_name}.pdf'))
