import math
import os
import argparse

from grakel.kernels import VertexHistogram, CoreFramework
from grakel.kernels import NeighborhoodSubgraphPairwiseDistance
from grakel.kernels import WeisfeilerLehman

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from Methods.SketchGraphs import SubSampleGraphs
from Utils.load_data import load_smi2mol
from Utils.s2m_utils import from_grkl_to_dict, to_networkx, eval_graph, plot_qm9


def compute_sketched_feature_maps(Y, Y_tr, output_kernel, Ry, Ay):
    KRy = Ry.multiply_Gram_one_side(Y_tr, output_kernel, Y=Y)
    Z = (KRy).dot(Ay)
    return Z


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken,
        d_model,
        nhead,
        d_hid,
        d_out,
        nlayers,
        dropout=0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_out)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # embedding of the first token - BERT version
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )[0]
        output = self.linear(output)
        return output


def Gaussian_kernel(gamma=None):
    """
    Defining Gaussian kernel
    """
    def Compute_Gram(X, Y=None):
        return rbf_kernel(X, Y, gamma=gamma)

    return Compute_Gram


def data_process(raw_text_iter, max_len, vocab, tokenizer):
    """
    Converts raw text into a flat Tensor.
    """
    pad_token = vocab["<pad>"]
    print(pad_token)
    tokens = [vocab(tokenizer(item)) for item in raw_text_iter]
    tokens_paded = []
    masks = []
    for item in tokens:
        if len(item) >= max_len:
            tokens_paded.append(item[:max_len])
            masks.append([False] * max_len)
        else:
            tokens_paded.append(item + [pad_token] * (max_len - len(item)))
            masks.append([False] * len(item) + [True] * (max_len - len(item)))

    src = torch.tensor(tokens_paded, dtype=torch.long)
    src_mask = torch.tensor(masks, dtype=torch.bool)
    return src, src_mask


def train(
    epoch,
    net,
    optimizer,
    loader_tr,
    prefix_file_name_w_dir,
    verbose=True,
    save=True,
    device=None,
):
    losses_tr = []
    loss_tr = 0
    for batch_idx, (data, mask, target) in enumerate(loader_tr):
        data = data.to(device).transpose(0, 1)
        mask = mask.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net(data, mask)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_tr += loss.item()
        if batch_idx % 10 == 0 and verbose:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(loader_tr.dataset),
                    100.0 * batch_idx / len(loader_tr),
                    loss.item(),
                )
            )

    loss_tr /= len(loader_tr.dataset)
    losses_tr.append(loss_tr)
    if save:
        with open(prefix_file_name_w_dir + "losses_tr.npy", "wb") as f:
            np.save(f, losses_tr)
    return losses_tr


def validate(
    net,
    loader_val,
    prefix_file_name_w_dir,
    x_validation,
    Y_gkl_tr,
    G_trgts_val,
    Z_c,
    n_jobs,
    verbose=True,
    save=True,
    device=None,
):
    losses_val = []
    ed_vals = []
    loss_val = 0
    for data, mask, target in loader_val:
        data = data.to(device).transpose(0, 1)
        mask = mask.to(device)
        target = target.to(device)
        output = net(data, mask)
        loss_val += F.mse_loss(output, target).item()  # sum up batch loss

    loss_val /= len(loader_val.dataset)
    losses_val.append(loss_val)
    if save:
        with open(prefix_file_name_w_dir + "losses_val.npy", "wb") as f:
            np.save(f, losses_val)
    if verbose:
        print("\nValidation set: Average loss: {:.6f}\n".format(loss_val))
    f_val = (
        net.forward(
            x_validation[0].to(device).transpose(
                0, 1), x_validation[1].to(device)
        )
        .detach()
        .cpu()
        .numpy()
    )
    scores_val = f_val.dot(Z_c.T)
    idx_pred_val = np.argmax(scores_val, axis=1)
    Y_gkl_pred_val = Y_gkl_tr[idx_pred_val]
    Y_dict_pred_val = from_grkl_to_dict(Y_gkl_pred_val)
    G_preds_val = [to_networkx(y_dict_pred) for y_dict_pred in Y_dict_pred_val]
    mean_eds_val = eval_graph(G_preds_val, G_trgts_val, n_jobs=n_jobs)[
        "edit_distance"]
    ed_vals.append(mean_eds_val)
    if save:
        with open(prefix_file_name_w_dir + "ed_vals.npy", "wb") as f:
            np.save(f, ed_vals)
    return losses_val, ed_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed_split", type=int, default=64)
    parser.add_argument("--output_kernel", type=str, default="CORE-WL")
    parser.add_argument("--data_path", type=str, default="Data/smi2mol/")
    parser.add_argument(
        "--mys_kernel",
        type=int,
        default=3200,
        help="Skteching size of the output kernel",
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of heads of Transformer encoder"
    )
    parser.add_argument(
        "--nlayers", type=int, default=6, help="Number of layers of Transformer encoder"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability of Transformer encoder",
    )
    parser.add_argument(
        "--dim", type=int, default=512, help="Hidden dimension of Transformer encoder"
    )

    args = parser.parse_args()

    # Setting random seed
    random_seed = args.random_seed_split
    np.random.seed(seed=random_seed)

    n_jobs = 16

    # Load and process dataset
    path = args.data_path
    delete_atoms = True
    (
        X_tr,
        Y_gkl_tr,
        Y_dict_tr,
        X_val,
        Y_gkl_val,
        Y_dict_val,
        X_te,
        Y_gkl_te,
        Y_dict_te,
    ) = load_smi2mol(
        path=path, n_valid=500, n_test=2000, delete_atoms=True, random_seed=random_seed
    )

    X_tr = [" ".join(list(item)) for item in X_tr]
    X_val = [" ".join(list(item)) for item in X_val]
    X_te = [" ".join(list(item)) for item in X_te]

    # Transform the input SMILES
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(tokenizer, X_tr), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    ntokens = len(vocab)  # size of vocabulary
    max_len = 25
    x_trai = data_process(X_tr, max_len, vocab, tokenizer)
    x_validation = data_process(X_val, max_len, vocab, tokenizer)
    x_te = data_process(X_te, max_len, vocab, tokenizer)
    n_tr = len(x_trai[0])
    print(f"Training datset size is {n_tr}")
    n_val = len(x_validation[0])
    print(f"Validation datset size is {n_val}")
    n_te = len(x_te[0])
    print(f"Test datset size is {n_te}")

    # Transform the output graph
    G_trgts_val = [to_networkx(y_dict_val) for y_dict_val in Y_dict_val]
    G_trgts = [to_networkx(y_dict_te) for y_dict_te in Y_dict_te]

    # Define output kernel
    output_kernel_list = {
        "NSPD": NeighborhoodSubgraphPairwiseDistance(n_jobs=n_jobs, normalize=True),
        "WL-VH": WeisfeilerLehman(
            n_jobs=n_jobs, normalize=True, base_graph_kernel=VertexHistogram
        ),
        "CORE-WL": CoreFramework(
            n_jobs=n_jobs, normalize=True, base_graph_kernel=WeisfeilerLehman
        ),
    }
    kernel_name = args.output_kernel
    output_kernel = output_kernel_list[kernel_name]
    my = args.mys_kernel

    # Define sketcher for output kernel
    output_sketch_name = "SubSample"
    Ry = SubSampleGraphs((my, n_tr))

    # Computer the basis
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
    ry = Ay.shape[1]  # dimension of the basis

    # Transform the output graphs to the sketched feature space
    Z_tr = compute_sketched_feature_maps(
        Y_gkl_tr, Y_gkl_tr, output_kernel, Ry, Ay)
    Z_val = compute_sketched_feature_maps(
        Y_gkl_val, Y_gkl_tr, output_kernel, Ry, Ay)
    Z_c = Z_tr.copy()

    # Get dataloder for pytorch
    batch_size = 64
    z_trai = torch.from_numpy(Z_tr).float()
    z_validation = torch.from_numpy(Z_val).float()
    dataset_tr = torch.utils.data.TensorDataset(*x_trai, z_trai)
    loader_tr = torch.utils.data.DataLoader(
        dataset_tr, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dataset_val = torch.utils.data.TensorDataset(*x_validation, z_validation)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Tarining g_W: solving the surrogate problem
    # define the input transformer model
    dim = args.dim
    nlayers = args.nlayers
    dropout = args.dropout
    nhead = args.nhead
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    prefix_file_name = "_".join(
        [
            "DSOKR",
            kernel_name,
            output_sketch_name,
            str(my),
            f"dim{dim}",
            f"nlayers{nlayers}",
            f"dropout{dropout}",
            f"nhead{nhead}",
        ]
    )
    save_dir = f"exper/s2m_seed{random_seed}/"
    prefix_file_name_w_dir = save_dir + prefix_file_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(prefix_file_name_w_dir + "_Zc", Z_c)

    net = TransformerModel(ntokens, dim, nhead, dim * 4, ry, nlayers, dropout).to(
        device
    )
    pytorch_total_params = sum(p.numel()
                               for p in net.parameters() if p.requires_grad)
    print(f"Total parameters of the NN is {pytorch_total_params}")

    # begin the optimization
    optimizer = optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )
    n_epochs = 50
    for epoch in tqdm(range(1, n_epochs + 1)):
        losses_tr = train(epoch, net, optimizer, loader_tr,
                          prefix_file_name_w_dir, device=device)
        losses_val, ed_vals = validate(net, loader_val, prefix_file_name_w_dir,
                                       x_validation, Y_gkl_tr, G_trgts_val, Z_c, n_jobs, device=device)
        torch.save(
            net.state_dict(
            ), os.path.join(save_dir, "model" + str(epoch) + ".pth")
        )

    plt.figure()
    plt.plot(losses_tr, label="train")
    plt.plot(losses_val, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix_file_name_w_dir + "losses.pdf", transparent=True)
    plt.close()

    plt.figure()
    plt.plot(ed_vals)
    plt.xlabel("epochs")
    plt.ylabel("Validation Edit Distance")
    plt.tight_layout()
    plt.savefig(prefix_file_name_w_dir + "ed_vals.pdf", transparent=True)
    plt.close()

    idx_best_model = np.argmin(ed_vals) + 1
    f = open(prefix_file_name_w_dir + "idx_best_model.txt", "w")
    f.write("Best epoch: " + str(idx_best_model) + "\n")
    f.close()

    ################# Get the performance of the best trained model ###########
    net = TransformerModel(ntokens, dim, nhead, dim * 4, ry, nlayers, dropout).to(
        device
    )

    net.load_state_dict(
        torch.load(
            os.path.join(save_dir, "model" + str(idx_best_model) + ".pth"))
    )
    torch.save(
        net.state_dict(),
        prefix_file_name_w_dir + "model" + str(idx_best_model) + ".pth",
    )

    # Get results on validation set 
    f_val = (
        net.forward(
            x_validation[0].to(device).transpose(0, 1),
            x_validation[1].to(device))
        .detach()
        .cpu()
        .numpy()
    )
    scores_val = f_val.dot(Z_c.T)
    idx_pred_val = np.argmax(scores_val, axis=1)
    Y_gkl_pred_val = Y_gkl_tr[idx_pred_val]
    Y_dict_pred_val = from_grkl_to_dict(Y_gkl_pred_val)
    G_preds_val = [to_networkx(y_dict_pred) for y_dict_pred in Y_dict_pred_val]
    G_trgts_val = [to_networkx(y_dict_val) for y_dict_val in Y_dict_val]
    
    f = open(prefix_file_name_w_dir + "eds.txt", "w")
    mean_eds_val = eval_graph(G_preds_val, G_trgts_val, n_jobs=n_jobs)[
        "edit_distance"]
    mean_eds_val_no_ef = eval_graph(
        G_preds_val, G_trgts_val, n_jobs=n_jobs, with_edge_feature=False
    )["edit_distance"]
    f.write("Validation mean edit distance: " + str(mean_eds_val) + "\n")
    f.write(
        "Validation mean edit distance w/o edge feature: "
        + str(mean_eds_val_no_ef)
        + "\n"
    )

    # Get results on test set 
    f_te = (
        net.forward(x_te[0].to(device).transpose(0, 1),
                    x_te[1].to(device))
        .detach()
        .cpu()
        .numpy()
    )
    scores = f_te.dot(Z_c.T)
    k = 5
    idx_topk = np.argsort(scores, axis=1)[:, -k:]
    Y_pred_topk_te = np.empty((n_te, 0))
    for i in range(k):
        Y_pred_i = Y_gkl_tr[idx_topk[:, -(i + 1)]].reshape((-1, 1)).copy()
        Y_pred_topk_te = np.hstack((Y_pred_topk_te, Y_pred_i))
    #scores_topk = np.flip(np.sort(scores, axis=1), axis=1)[:, :k]
    Y_dict_pred_te = from_grkl_to_dict(Y_pred_topk_te[:, 0])
    G_preds = [to_networkx(y_dict_pred) for y_dict_pred in Y_dict_pred_te]
    G_trgts = [to_networkx(y_dict_te) for y_dict_te in Y_dict_te]

    res = eval_graph(G_preds, G_trgts, n_jobs=n_jobs)
    eds = res["eds"]
    mean_eds = res["edit_distance"]
    mean_eds_no_ef = eval_graph(
        G_preds, G_trgts, n_jobs=n_jobs, with_edge_feature=False
    )["edit_distance"]
    f.write("Test mean edit distance: " + str(mean_eds) + "\n")
    f.write("Test mean edit distance w/o edge feature: " +
            str(mean_eds_no_ef) + "\n")
    f.close()

    ################# Draw serveral predicted examples #########################
    i = 0
    for G_pred, G_tgt, edist in zip(G_preds, G_trgts, eds):
        if i >= 20:
            break
        fig, axs = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1))
        plot_qm9(G_pred, axs, f"GED w/o EF = {edist}", draw_edge_feature=True)
        fig.tight_layout()
        fig.savefig(os.path.join(prefix_file_name_w_dir, f"dsokr_pred{i}.pdf"))

        fig, axs = plt.subplots(1, 1, figsize=(3 * 1, 3 * 1))
        plot_qm9(G_tgt, axs, draw_edge_feature=True)
        fig.tight_layout()
        fig.savefig(os.path.join(prefix_file_name_w_dir, f"dsokr_tgt{i}.pdf"))
        i += 1
