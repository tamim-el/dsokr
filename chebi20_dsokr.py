import os
import argparse
import random

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertModel
from transformers.optimization import get_linear_schedule_with_warmup
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from Methods.Sketch import SubSample, pSparsified, Gaussian
from Utils.load_data import load_text2mols
from Utils.chebi_20_utils import gaussian_kernel


def compute_sketched_feature_maps(Y, Y_tr, output_kernel, Ry, Ay):
    KRy = Ry.multiply_Gram_one_side(Y, output_kernel, Y=Y_tr)
    Z = (KRy).dot(Ay)
    return Z


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_out,
    ):
        super().__init__()
        self.model_type = "BERT"
        self.transformer_model = BertModel.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        d_model = self.transformer_model.config.hidden_size
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, src, src_key_padding_mask):
        output = self.transformer_model(src, attention_mask=src_key_padding_mask)[
            "pooler_output"
        ]
        output = self.linear(output)
        return output


def data_process(raw_text_iter, max_len, tokenizer):
    srcs = []
    src_masks = []
    for sentence in raw_text_iter:
        """Converts raw text into a flat Tensor."""
        tokenized = tokenizer(
            sentence, truncation=True, max_length=max_len, padding="max_length"
        )

        src = tokenized["input_ids"]
        src_mask = tokenized["attention_mask"]
        srcs.append(src)
        src_masks.append(src_mask)

    return torch.tensor(srcs), torch.tensor(src_masks)


def train(epoch, net, loader_tr, device, verbose=True):
    loss_tr = 0
    for batch_idx, (data, mask, target) in enumerate(loader_tr):
        data = data.to(device)
        mask = mask.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net(data, mask)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
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
    return loss_tr


def validate(net, loader_val, device, verbose=True):
    net.eval()
    loss_val = 0
    f_val_batch = []
    for data, mask, target in loader_val:
        data = data.to(device)
        mask = mask.to(device)
        target = target.to(device)
        output = net(data, mask)
        f_val_batch.append(output.detach().cpu().numpy())
        loss_val += F.mse_loss(output, target).item()  # sum up batch loss

    loss_val /= len(loader_val.dataset)
    net.train()

    if verbose:
        print("\nValidation set: Average loss: {:.6f}\n".format(loss_val))
    f_val = np.concatenate(f_val_batch, axis=0)
    scores_val = f_val.dot(Z_c.T)

    res_val = eval_rank(scores_val, offset=len(Z_tr))

    return loss_val, res_val["mrr"]


def eval_rank(scores, offset):
    cid_locs = np.argsort(scores, axis=1)[:, ::-1]
    ranks_tmps = np.argsort(cid_locs)
    ranks = []
    for i, item in enumerate(ranks_tmps):
        ranks.append(item[offset + i] + 1)

    ranks = np.array(ranks)

    return {
        "ranks": ranks,
        "mrr": np.mean(1 / ranks),
        "top1": np.mean(ranks <= 1),
        "top10": np.mean(ranks <= 10),
        "ranks_all": ranks_tmps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my", type=int, default=100)
    parser.add_argument("--output_sketch_name", type=str, default="SubSample")
    parser.add_argument("--output_kernel", type=str, default="gaussian")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default='Data/chebi-20/')
    parser.add_argument("--save_dir", type=str, default='text2mol')
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_path = os.path.join(args.save_dir, 'torch_models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Setting random seed
    np.random.seed(seed=args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    ################# Load and pre-processing data ##########
    (
        X_tr,
        Y_dict_tr,
        X_val,
        Y_dict_val,
        X_te,
        Y_dict_te,
        Y_dict_cand,
    ) = load_text2mols(path=args.data_dir)

    Y_tr = np.array([item['mol2vec'] for item in Y_dict_tr])
    Y_val = np.array([item['mol2vec'] for item in Y_dict_val])
    Y_te = np.array([item['mol2vec'] for item in Y_dict_te])
    Y_cand = np.array([item['mol2vec'] for item in Y_dict_cand])

    max_len = 256
    text_tokenizer = BertTokenizerFast.from_pretrained(
        "allenai/scibert_scivocab_uncased")
    x_te = data_process(X_te, max_len, text_tokenizer)
    x_trai = data_process(X_tr, max_len, text_tokenizer)
    x_validation = data_process(X_val, max_len, text_tokenizer)
    n_tr = len(x_trai[0])
    print(n_tr)
    n_val = len(x_validation[0])
    print(n_val)
    n_te = len(x_te[0])
    print(n_te)

    my = args.my
    output_sketch_name = args.output_sketch_name

    if output_sketch_name == "SubSample":
        Ry = SubSample((my, n_tr))
    elif output_sketch_name == "Gaussian":
        Ry = Gaussian((my, n_tr))
    elif output_sketch_name == 'pSparsified':
        p = 20.0 / n_tr
        Ry = pSparsified((my, n_tr), p=p, type='Rademacher')
    else:
        raise ValueError

    # muy = 1e-8

    if args.output_kernel == "gaussian":
        output_kernel = gaussian_kernel(args.gamma)
    elif args.output_kernel == "cosine":
        output_kernel = cosine_similarity
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
    print(ry)

    Z_tr = compute_sketched_feature_maps(
        Y_tr, Y_tr, output_kernel, Ry, Ay
    )
    Z_val = compute_sketched_feature_maps(
        Y_val, Y_tr, output_kernel, Ry, Ay
    )
    Z_c = compute_sketched_feature_maps(
        Y_cand, Y_tr, output_kernel, Ry, Ay
    )
    print(Z_c.shape)

    z_trai = torch.from_numpy(Z_tr).float()
    z_validation = torch.from_numpy(Z_val).float()

    batch_size = 32
    dataset_tr = torch.utils.data.TensorDataset(*x_trai, z_trai)
    loader_tr = torch.utils.data.DataLoader(
        dataset_tr, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dataset_val = torch.utils.data.TensorDataset(*x_validation, z_validation)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=0
    )
    dataset_te = torch.utils.data.TensorDataset(*x_te)
    loader_te = torch.utils.data.DataLoader(
        dataset_te, batch_size=batch_size, shuffle=False, num_workers=0
    )

    prefix_file_name = "_".join(
        [
            "DSOKR",
            args.output_kernel,
            output_sketch_name,
            str(my),
            str(args.gamma),
            str(args.random_seed),
        ]
    )

    prefix_file_name_w_dir = f"{args.save_dir}/" + prefix_file_name

    np.save(prefix_file_name_w_dir + 'Z_c.npy', Z_c)

    ################# Model Training (Solving the regression problem) ##########
    # Create Neural Network model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = TransformerModel(ry).to(device)
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print(f"Total parameters of the NN is {pytorch_total_params}")

    lr = 3e-5
    n_epochs = 50
    num_warmup_steps = 1000

    optimizer = optim.Adam(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )

    n_train_steps = n_epochs * len(loader_tr) - num_warmup_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=n_train_steps,
    )

    losses_tr = []
    losses_val = []
    mrrs_val = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss_tr = train(epoch, net, loader_tr, device)
        losses_tr.append(loss_tr)
        loss_val, mrr_val = validate(net, loader_val, device)
        losses_val.append(loss_val)
        mrrs_val.append(mrr_val)
        torch.save(
            net.state_dict(),
            f"{args.save_dir}/torch_models/model" + str(epoch) + ".pth",
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
    plt.plot(mrrs_val)
    plt.xlabel("epochs")
    plt.ylabel("Validation MRR")
    plt.tight_layout()
    plt.savefig(prefix_file_name_w_dir + "mrr_vals.pdf", transparent=True)
    plt.close()

    idx_best_model = np.argmax(mrrs_val) + 1
    f = open(prefix_file_name_w_dir + "idx_best_model.txt", "w")
    f.write("Best epoch: " + str(idx_best_model) + "\n")
    f.close()

    ################# Get the performance of the trained model ##########
    net = TransformerModel(ry).to(device)

    # Load best checkpoint
    net.load_state_dict(
        torch.load(f"{args.save_dir}/torch_models/model" +
                   str(idx_best_model) + ".pth")
    )
    torch.save(
        net.state_dict(),
        prefix_file_name_w_dir + "model" + str(idx_best_model) + ".pth",
    )

    # Get the performance on the validation set
    f_val_batch = []
    for data, mask, target in loader_val:
        data = data.to(device)
        mask = mask.to(device)
        target = target.to(device)
        output = net(data, mask)
        f_val_batch.append(output.detach().cpu().numpy())

    f_val = np.concatenate(f_val_batch, axis=0)

    scores_val = f_val.dot(Z_c.T)
    res_val = eval_rank(scores_val, offset=len(Z_tr))

    f = open(prefix_file_name_w_dir + "ranking.txt", "w")
    f.write("Validation mrr: " + str(res_val["mrr"]) + "\n")
    f.write("Validation top1: " + str(res_val["top1"]) + "\n")
    f.write("Validation top10: " + str(res_val["top10"]) + "\n")

    np.save(prefix_file_name_w_dir + "rank_valid.npy", res_val['ranks_all'])

    # Get the performance on the test set
    f_te_batch = []
    for data, mask in loader_te:
        data = data.to(device)
        mask = mask.to(device)
        target = target.to(device)
        output = net(data, mask)
        f_te_batch.append(output.detach().cpu().numpy())

    f_te = np.concatenate(f_te_batch, axis=0)
    scores_te = f_te.dot(Z_c.T)
    res_te = eval_rank(scores_te, offset=len(Z_tr) + len(Z_val))

    f.write("Test mrr: " + str(res_te["mrr"]) + "\n")
    f.write("Test top1: " + str(res_te["top1"]) + "\n")
    f.write("Test top10: " + str(res_te["top10"]) + "\n")
    f.close()

    np.save(prefix_file_name_w_dir + "rank_test.npy", res_te['ranks_all'])
