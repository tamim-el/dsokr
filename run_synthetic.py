import os

from tempfile import TemporaryDirectory

import numpy as np
from time import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error

# sketch
from Methods.Sketch import SubSample, Gaussian


from Utils.load_data import load_toy_DSOKR
from Utils.nets import Net1

import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42})

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')




# Defining linear kernel
def Linear_kernel():
    def Compute_Gram(X, Y=None):
        return linear_kernel(X, Y)
    return Compute_Gram


def compute_sketched_feature_maps(Y, Y_tr, output_kernel, Ry, Ay):
        
    KRy = Ry.multiply_Gram_one_side(Y, output_kernel, Y=Y_tr)
    Z = (KRy).dot(Ay)
    return Z



# Setting random seed
np.random.seed(seed=42)



# Loading dataset
n = 65000
dx = 2000
dy = 1000
dytrue = 50
r = 0.5
sigma = 0.01

n_rep = 5

X, Y, _ = load_toy_DSOKR(n=n, dx=dx,
                dy=dy, dytrue=dytrue,
                r=r, sigma=sigma)

n_tr = 50000
n_te = 10000
n_val = 5000
input_dim = X.shape[1]
output_dim = Y.shape[1]

X_tr, Y_tr = X[:n_tr], Y[:n_tr]
X_val, Y_val = X[n_tr : n_tr + n_val], Y[n_tr : n_tr + n_val]
X_te, Y_te = X[n_tr + n_val:], Y[n_tr + n_val:]

prefix_file_name = 'NN_dytrue' + str(dytrue) + '_' \
            + 'sigma' + str(sigma) + '_' + 'rep' + str(n_rep) + '_'



###### Leverage scores ################################################################

def approx_lev_scores(Y, kernel, p, m, L):
    n = Y.shape[0]
    indices = np.random.choice(n, m, replace=True, p=p)
    Y_sampled = Y[indices]
    W = kernel(Y_sampled)
    C = kernel(Y, Y_sampled)
    V, D, _ = np.linalg.svd(W)
    VDsqrt = (D ** (-1/2)) * V
    B = C.dot(VDsqrt)
    M = B.T.dot(B) + n * L * np.eye(m)
    Minv = np.linalg.inv(M)
    lev = np.zeros(n)
    for i in range(n):
        lev[i] = B[i, :].reshape((1, -1)).dot(Minv).dot(B[i, :].reshape((-1, 1)))
    return lev


output_kernel = Linear_kernel()

m = int(np.sqrt(n_tr))
p = (1 / n_tr) * np.ones(n_tr)
L = 1e-4


lev = approx_lev_scores(Y_tr, output_kernel, p, m, L)


fig, ax = plt.subplots(figsize=(15, 12))
ax.plot(np.sort(lev)[::-1][:400], lw=10)
plt.ylabel('ALS', fontsize=50)
plt.xlabel('Sorted entries', fontsize=50)
ax.set_xticklabels([str(i)[:3] for i in ax.get_xticks()], fontsize = 50)
ax.set_yticklabels([str(i)[:5] for i in ax.get_yticks()], fontsize = 50)
plt.tight_layout()
plt.savefig('Plots/synthetic_lev_scores.pdf', transparent=True)
plt.close()



########### One-layer neural network ###############################################""

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


X_te = torch.from_numpy(X_te).float()
X_te_cuda = X_te.to(device)

x_trai = torch.from_numpy(X_tr).float()
y_trai = torch.from_numpy(Y_tr).float()
x_validation = torch.from_numpy(X_val).float()
y_validation = torch.from_numpy(Y_val).float()
x_validation_cuda = x_validation.to(device)



batch_size = 32


def train_normal(epoch, net, verbose=True):
        
    loss_tr = 0

    for batch_idx, (data, target) in enumerate(loader_tr):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_tr += loss.item()
        if batch_idx % 10 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_tr.dataset),
                100. * batch_idx / len(loader_tr), loss.item()))

    loss_tr /= len(loader_tr.dataset)
    losses_tr.append(loss_tr)

def validate_normal(net, verbose=True):

    loss_val = 0
    
    for data, target in loader_val:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss_val += F.mse_loss(output, target).item() # sum up batch loss

    loss_val /= len(loader_val.dataset)
    losses_val.append(loss_val)
    if verbose:
        print('\nValidation set: Average loss: {:.6f}\n'.format(
        loss_val))


print('One-layer neural network in process...')


num_workers = 4


dataset_tr = torch.utils.data.TensorDataset(x_trai, y_trai)
loader_tr = torch.utils.data.DataLoader(dataset_tr,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)

dataset_val = torch.utils.data.TensorDataset(x_validation, y_validation)
loader_val = torch.utils.data.DataLoader(dataset_val,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)

net = Net1(dim_inputs=input_dim, dim_outputs=output_dim).to(device)

lr = 1e-3

optimizer = optim.Adam(
                net.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)


n_epochs = 20

losses_tr = []
losses_val = []

best_val_loss = 1e9

# Create a temporary directory to save training checkpoints
with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, prefix_file_name + 'best_model_params.pt')
    for epoch in range(1, n_epochs + 1):
        train_normal(epoch, net)
        validate_normal(net)
        eval_val = losses_val[-1]
        if best_val_loss > eval_val:
            best_val_loss = eval_val
            idx_best_model = epoch
            torch.save(net.state_dict(), best_model_params_path)
    net.load_state_dict(torch.load(best_model_params_path))



Y_pred_val = net.forward(x_validation_cuda).detach().cpu().numpy()
mse_val_NN = mean_squared_error(Y_pred_val, Y_val)

Y_pred_te = net.forward(X_te_cuda).detach().cpu().numpy()
mse_te_NN = mean_squared_error(Y_pred_te, Y_te)





######## DSOKR models #######################################################################


def train(epoch, net, verbose=True):
        
    loss_tr = 0

    for batch_idx, (data, target) in enumerate(loader_tr):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_tr += loss.item()
        if batch_idx % 10 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_tr.dataset),
                100. * batch_idx / len(loader_tr), loss.item()))

    loss_tr /= len(loader_tr.dataset)
    losses_tr.append(loss_tr)

def validate(net, verbose=True):

    loss_val = 0
    
    for data, target in loader_val:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss_val += F.mse_loss(output, target).item() # sum up batch loss

    loss_val /= len(loader_val.dataset)
    losses_val.append(loss_val)
    if verbose:
        print('\nValidation set: Average loss: {:.6f}\n'.format(
        loss_val))
    f_val = net.forward(x_validation_cuda).detach().cpu().numpy()
    Y_pred_val = f_val.dot(Ay.T).dot(RyY)
    mse_val = mean_squared_error(Y_pred_val, Y_val)
    mse_validations.append(mse_val)


output_sketch_names = ['SubSample', 'Gaussian']
mys = [2, 5, 10, 15, 20, 25,
    30, 40, 50, 75, 100, 125,
    150, 175, 200, 250, 300,
    350, 400]
n_mys = len(mys)

mse_perfecth_vals_SubS = np.zeros((n_mys, n_rep))
mse_perfecth_vals_Gaus = np.zeros((n_mys, n_rep))

mse_tes_SubS = np.zeros((n_mys, n_rep))
mse_tes_Gaus = np.zeros((n_mys, n_rep))

for i_osn, output_sketch_name in enumerate(output_sketch_names):

    mse_perfecth_vals = np.zeros((n_mys, n_rep))

    mse_vals = np.zeros((n_mys, n_rep))
    mse_tes = np.zeros((n_mys, n_rep))

    prefix_file_name_perfecth = 'Perfecth_dytrue' + str(dytrue) + '_' \
            + 'sigma' + str(sigma) + '_' + 'rep' + str(n_rep) + '_' \
                + output_sketch_name + '_'

    prefix_file_name = 'DSOKR_dytrue' + str(dytrue) + '_' \
            + 'sigma' + str(sigma) + '_' + 'rep' + str(n_rep) + '_' \
                + output_sketch_name + '_'

    for i_my, my in enumerate(mys):

        for i_rep in range(n_rep):

            if i_osn == 0:
                Ry = SubSample((my, n_tr))
            else:
                Ry = Gaussian((my, n_tr))

            RyKRyT = Ry.multiply_Gram_both_sides(Y_tr, output_kernel)
            V, D, _ = np.linalg.svd(RyKRyT)
            nnz_D = np.logical_not(np.isclose(D, np.zeros(D.shape), rtol=1e-12,))
            D_r, V_r = D[nnz_D], V[:, nnz_D]
            Ay = (D_r ** (-1/2)) * V_r

            ry = Ay.shape[1]

            Z_tr = compute_sketched_feature_maps(Y_tr, Y_tr, output_kernel, Ry, Ay)
            Z_val = compute_sketched_feature_maps(Y_val, Y_tr, output_kernel, Ry, Ay)

            z_trai = torch.from_numpy(Z_tr).float()
            z_validation = torch.from_numpy(Z_val).float()

            RyY = Ry.multiply_matrix_one_side(Y_tr, right=False)

            dataset_tr = torch.utils.data.TensorDataset(x_trai, z_trai)
            loader_tr = torch.utils.data.DataLoader(dataset_tr,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)

            dataset_val = torch.utils.data.TensorDataset(x_validation, z_validation)
            loader_val = torch.utils.data.DataLoader(dataset_val,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)

            net = Net1(dim_inputs=input_dim, dim_outputs=ry).to(device)

            lr = 1e-3

            optimizer = optim.Adam(
                            net.parameters(),
                            lr=lr,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0,
                            amsgrad=False)


            n_epochs = 20

            losses_tr = []
            losses_val = []
            mse_validations = []

            best_val_loss = 1e9

            # Create a temporary directory to save training checkpoints
            with TemporaryDirectory() as tempdir:
                best_model_params_path = os.path.join(tempdir, prefix_file_name + 'best_model_params.pt')
                for epoch in range(1, n_epochs + 1):
                    train(epoch, net, save=False)
                    validate(net, save=False)
                    eval_val = mse_validations[-1]
                    if best_val_loss > eval_val:
                        best_val_loss = eval_val
                        idx_best_model = epoch
                        torch.save(net.state_dict(), best_model_params_path)
                net.load_state_dict(torch.load(best_model_params_path))


            Y_pred_perfecth_val = Z_val.dot(Ay.T).dot(RyY)
            mse_val = mean_squared_error(Y_pred_val, Y_val)
            mse_perfecth_vals[i_my, i_rep] = mse_val

            f_val = net.forward(x_validation_cuda).detach().cpu().numpy()
            Y_pred_val = f_val.dot(Ay.T).dot(RyY)
            mse_val = mean_squared_error(Y_pred_val, Y_val)
            mse_vals[i_my, i_rep] = mse_val


            f_te = net.forward(X_te_cuda).detach().cpu().numpy()
            Y_pred_te = f_te.dot(Ay.T).dot(RyY)
            mse_te = mean_squared_error(Y_pred_te, Y_te)
            mse_tes[i_my, i_rep] = mse_te

    if i_osn == 0:
        mse_perfecth_vals_SubS = mse_perfecth_vals
        mse_tes_SubS = mse_tes
    else:
        mse_perfecth_vals_Gaus = mse_perfecth_vals
        mse_tes_Gaus = mse_tes


mse_val_perfecth_SubS_mean = np.mean(mse_perfecth_vals_SubS, axis=1)
mse_val_perfecth_SubS_std = np.std(mse_perfecth_vals_SubS, axis=1)

mse_val_perfecth_Gaus_mean = np.mean(mse_perfecth_vals_Gaus, axis=1)
mse_val_perfecth_Gaus_std = np.std(mse_perfecth_vals_Gaus, axis=1)

fig, ax = plt.subplots(figsize=(15, 12))
ax.errorbar(mys, mse_val_perfecth_SubS_mean,
            yerr=mse_val_perfecth_SubS_std,
            fmt='-o', label='Sub-Sample', lw=2, elinewidth=2)
ax.errorbar(mys, mse_val_perfecth_Gaus_mean,
            yerr=mse_val_perfecth_Gaus_std,
            fmt='-*', label='Gaussian', lw=2, elinewidth=2)
plt.axhline(0, linewidth=2, c='k')
ax.legend(markerscale=3)
plt.setp(ax.get_legend().get_texts(), fontsize=50)
plt.ylabel('Validation Perfect h MSE', fontsize=50)
plt.xlabel('$m$', fontsize=40)
ax.set_xticklabels([str(i)[:3] for i in ax.get_xticks()], fontsize = 50)
ax.set_yticklabels([str(i) for i in ax.get_yticks()], fontsize = 50)
plt.tight_layout()
plt.savefig('Plots/synthetic_perfecth.pdf', transparent=True)
plt.close()


mse_NN = mse_te_NN * np.ones_like(mys)

mse_te_SubS_diff = mse_tes_SubS - mse_te_NN

mse_te_SubS_diff_mean = np.mean(mse_te_SubS_diff, axis=1)
mse_te_SubS_diff_std = np.std(mse_te_SubS_diff, axis=1)

mse_te_Gaus_diff = mse_tes_Gaus - mse_te_NN

mse_te_Gaus_diff_mean = np.mean(mse_te_Gaus_diff, axis=1)
mse_te_Gaus_diff_std = np.std(mse_te_Gaus_diff, axis=1)


fig, ax = plt.subplots(figsize=(15, 12))
ax.errorbar(mys, mse_te_SubS_diff_mean,
            yerr=mse_te_SubS_diff_std,
            fmt='-o', label='Sub-Sample', lw=2, elinewidth=2)
ax.errorbar(mys, mse_te_Gaus_diff_mean,
            yerr=mse_te_Gaus_diff_std,
            fmt='-*', label='Gaussian', lw=2, elinewidth=2)
plt.axhline(0, linewidth=2, c='k')
ax.legend(markerscale=3)
plt.setp(ax.get_legend().get_texts(), fontsize=50)
plt.ylabel('Test DSOKR MSE - NN MSE', fontsize=50)
plt.xlabel('$m$', fontsize=40)
ax.set_xticklabels([str(i)[:3] for i in ax.get_xticks()], fontsize = 50)
ax.set_yticklabels([str(i) for i in ax.get_yticks()], fontsize = 50)
plt.tight_layout()
plt.savefig('Plots/synthetic_mse.pdf', transparent=True)
plt.close()