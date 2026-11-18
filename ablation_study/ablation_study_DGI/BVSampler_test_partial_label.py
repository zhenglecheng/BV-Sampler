#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import typing
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from sklearn.metrics import f1_score as F1
import os
from model import OptISGCN, HistoricalNormCache, InfoNCELoss

# --dataset PubMed --norm_feat true --fast true --hidden_dim 16 --init_batch 256 --sample_size 400 --early_stop 10 --wd 5e-4 --seed 42 --label_rate 0.05
# --dataset Reddit --fast true --norm_feat false --hidden_dim 128 --init_batch 1024 --sample_size 5120 --early_stop 100 --wd 1e-4 --seed 45 --label_rate 0.05
# --dataset ogbn-arxiv --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 1000 --lr 0.001 --label_rate 0.05

# --dataset PubMed --norm_feat true --fast true --hidden_dim 16 --init_batch 256 --sample_size 400 --wd 5e-4 --seed 42 --label_rate 0.05 --con_weight 0.1 --samp_inference true
# --dataset Reddit --con_weight 0.1 --fast true --norm_feat false --early_stop 100 --epoch 1000 --hidden_dim 128 --init_batch 1024 --sample_size 5120 --wd 1e-4 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000
# --dataset ogbn-arxiv --init_batch 1024 --sample_size 10240 --early_stop 300 --wd 1e-4  --batch_norm true --hidden_dim 256 --num_layers 2 --drop 0.5 --scale_factor 5 --epochs 2000 --lr 0.001 --con_weight 0 --label_rate 0.05 --samp_inference true --inference_init_batch 5000  --inference_sample_size 50000
# worse performance on arxiv when seed=43

def compose_mixed_batch(
    labeled_indices  : np.ndarray,
    unlabeled_indices: np.ndarray,
    batch_size       : int,
    labeled_ratio    : float,
    rng              : np.random.Generator,
) -> tuple:
    """
    Build one mini-batch with a fixed labeled / unlabeled split.

    When unlabeled_indices is empty the entire batch is labeled.

    Parameters
    ----------
    labeled_ratio : p ∈ (0, 1] — fraction of the batch drawn from labeled nodes
                    (default 0.80 → 80 % labeled, 20 % unlabeled)

    Returns
    -------
    labeled_batch   : (n_lab,)  node indices from the labeled pool
    unlabeled_batch : (n_unlab,) node indices from the unlabeled pool
                      (empty array when no unlabeled nodes exist)
    init_batch      : concatenation [labeled_batch | unlabeled_batch]
                      — this is passed directly to forward() as init_batch
    """
    has_unlabeled = len(unlabeled_indices) > 0

    if not has_unlabeled:
        n_lab   = min(batch_size, len(labeled_indices))
        lab_bat = rng.choice(labeled_indices, size=n_lab, replace=False)
        return lab_bat, np.array([], dtype=np.int64), lab_bat

    # n_lab   = max(1, int(round(batch_size * labeled_ratio)))
    # n_unlab = batch_size - n_lab
    #
    # n_lab   = min(n_lab,   len(labeled_indices))
    # n_unlab = min(n_unlab, len(unlabeled_indices))
    #
    # lab_bat   = rng.choice(labeled_indices,   size=n_lab,   replace=False)
    # unlab_bat = rng.choice(unlabeled_indices, size=n_unlab, replace=False)

    n_unlab   = max(1, int(round(batch_size * (1 - labeled_ratio))))
    lab_bat   = rng.choice(labeled_indices, size=batch_size, replace=False)
    unlab_bat = rng.choice(unlabeled_indices, size=n_unlab, replace=False)

    init_bat  = np.concatenate([lab_bat, unlab_bat])

    return lab_bat, unlab_bat, init_bat

def set_seeds(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed=seed)


def model_train(
    model            : OptISGCN,
    opt              : torch.optim.Optimizer,
    x                : torch.Tensor,
    y                : torch.Tensor,
    csr_mat          : sp.csr_matrix,
    training_mask    : torch.Tensor,
    loss_fn          : nn.Module,
    stoch            : bool,
    losses           : list,
    batch_list       : list,
    labeled_indices  : np.ndarray,
    unlabeled_indices: np.ndarray,
    labeled_ratio    : float,
    rng              : np.random.Generator,
    # infonce_loss     : InfoNCELoss,
    con_weight       : float,
    device           : torch.device,
) -> list:
    """
    One training step.

    Loss = L_CE(labeled_batch)  +  λ_c · L_InfoNCE(unlabeled_batch)

    The labeled and unlabeled portions of init_batch are always at known
    offsets (labeled first, unlabeled after), making it trivial to split
    the logits returned by forward() for the CE loss.

    The InfoNCE loss is computed by calling forward_embed() *twice* on the
    unlabeled nodes, producing two distinct neighbourhood views from
    independent sampler draws — no separate augmentation needed.
    """
    model.train()
    opt.zero_grad()

    # ── 1. Compose mixed batch ─────────────────────────────────────────────
    if stoch:
        # lab_bat, unlab_bat, init_bat = compose_mixed_batch(
        #     labeled_indices   = labeled_indices,
        #     unlabeled_indices = unlabeled_indices,
        #     batch_size        = batch_list[0],
        #     labeled_ratio     = labeled_ratio,
        #     rng               = rng,
        # )
        lab_bat, unlab_bat, init_bat = compose_mixed_batch(
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_list[0],
            labeled_ratio=labeled_ratio,  # still passed, now ignored internally
            rng=rng,
            # norm_cache=model.norm_cache,  # <-- add this
        )
    else:
        init_bat  = None
        lab_bat   = None
        unlab_bat = np.array([], dtype=np.int64)

    # ── 2. Forward pass ────────────────────────────────────────────────────
    # Determine whether InfoNCE is needed this step so we can request the
    # contrastive subgraph from forward() in the same sampling call.
    _need_contrastive = stoch and len(unlab_bat) > 1 and con_weight > 0.0
    out, contra_h, _, nce_subgraph = model(
        x                 = x,
        csr_mat           = csr_mat,
        drop              = True,
        stochastic        = stoch,
        batch_sizes       = batch_list,
        init_batch        = init_bat,
        contrastive_batch = unlab_bat if _need_contrastive else None,
    )

    # ── 3. Supervised CE loss on labeled nodes ─────────────────────────────
    if stoch:
        # Labeled nodes occupy the first len(lab_bat) rows of `out`
        n_lab  = len(lab_bat)
        labels = y[lab_bat]
        l_ce   = loss_fn(out[:n_lab], labels)
    else:
        l_ce = loss_fn(out[training_mask], y[training_mask])
        # l_ce = loss_fn(out[lab_bat], y[lab_bat])

    total_loss = l_ce

    # ── 4. InfoNCE loss on unlabeled nodes (if any) ────────────────────────
    if _need_contrastive:
        # z1 comes from the GCN forward pass already done above — no extra work.
        # z2 uses the pre-sampled NCE subgraph returned by forward(), so
        # _get_biased_subgraphs is NOT called a second time.
        z1 = model.proj_head(contra_h)
        adj_out_nce, alpha_out_nce = nce_subgraph
        z2       = model.forward_embed_presampled(adj_out_nce, alpha_out_nce, drop=True)
        l_con    = con_weight * model.infoNCELoss(z1, z2)
        total_loss = total_loss + l_con

    # ── 5. Backward ────────────────────────────────────────────────────────
    total_loss.backward()
    opt.step()
    losses.append(total_loss.item())
    return losses


#
# # Declare a training function
# def train(model: nn.Module, opt: torch.optim, X: torch.Tensor, y: torch.Tensor,
#         csr_mat: sp.csr_matrix,
#           training_mask: torch.Tensor, loss_function: nn.Module, stoch: bool, losses: list,
#           batch_list: list = None, training_indices: list = None) -> list:
#
#     # Set to train
#     model.train()
#     opt.zero_grad()
#
#     # Forward pass
#     out, init_batch = model(x=X, csr_mat=csr_mat, drop=False, stochastic=stoch,
#                             batch_sizes=batch_list, possible_training_nodes=training_indices)
#
#     # Make sure we are only grabbing training nodes computed from the FIRST batch
#     if stoch:
#         # Make training mask actually feasible
#         l = loss_function(out, y[init_batch])
#     else:
#         l = loss_function(out[training_mask], y[training_mask])
#
#     # Backward pass
#     l.backward()
#     opt.step()
#
#     # Save the losses
#     losses.append(l.item())
#
#     return losses


# Declare a training function
def sample_validation_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                            csr_mat: sp.csr_matrix,
                            batch_list: list, num_samp_inf: int, validation_mask: list,
                            loss_function: nn.Module, val_losses: list) -> list:

    # Set to eval
    model.eval()
    v_loss = 0

    c = 0
    for i in range(0, len(validation_mask), batch_list[0]):

        # Get this subset of nodes to perform prediction on
        m = validation_mask[i:i + batch_list[0]]

        # Forward pass
        out, res, init_batch = model.sample_predict(x=X, csr_mat=csr_mat,
                                                    init_batch=m, batch_sizes=batch_list,
                                                    num_inference_times=num_samp_inf)

        # Save loss
        v_loss += loss_function(res, y[init_batch]).detach().item()
        c += 1

    # Save the losses
    val_losses.append(v_loss / c)

    return val_losses


# Declare a training function
def validation_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, csr_mat: sp.csr_matrix,
          validation_mask: torch.Tensor, loss_function: nn.Module, val_losses: list) -> list:

    # Set to eval
    model.eval()

    # Get the loss
    _, out = model.predict(X, csr_mat)
    l = loss_function(out[validation_mask], y[validation_mask]).detach()

    # Save the losses
    val_losses.append(l.item())

    return val_losses


# Declare a testing function
def sample_test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, csr_mat: sp.csr_matrix,
            batch_list: list, num_samp_inf: int, mask: list, accuracy: list) -> list:

    # Set to test
    model.eval()

    # Loop over all of the indices
    total_out = np.array([])
    total_y = np.array([])
    for i in range(0, len(mask), batch_list[0]):

        # Get this subset of nodes to perform prediction on
        m = mask[i:i + batch_list[0]]

        # Forward pass
        out, res, init_batch = model.sample_predict(x=X, csr_mat=csr_mat,
                            init_batch=m, batch_sizes=batch_list, num_inference_times=num_samp_inf)

        total_out = np.concatenate((total_out, out.cpu().numpy())).flatten()
        total_y = np.concatenate((total_y, y[init_batch].cpu().numpy())).flatten()

    # Save the accuracy
    accuracy.append(F1(total_y, total_out, average='micro') * 100.0)

    return accuracy


# Declare a testing function
def test(model: nn.Module, X: torch.Tensor, y: torch.Tensor, csr_mat: sp.csr_matrix,
          mask: torch.Tensor, accuracy: list) -> list:

    # Set to test
    model.eval()

    # Forward pass
    out, _ = model.predict(X, csr_mat)
    out = out[mask].cpu().numpy()
    y = y[mask].cpu().numpy()
    acc = F1(y, out, average='micro') * 100.0

    # Save the accuracy
    accuracy.append(acc)

    return accuracy


# Declare some helper functions
def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    Copied from: https://github.com/matenure/FastGCN/blob/b8e6e6412d8cb696e0adb7910832ed1ae7243ac9/utils.py#L136
    """

    # Convert to COO
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)

    # Get the degree
    rowsum = np.array(adj.sum(1))

    # Compute the normalizing factor
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Return CSR matrix
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


# Declare some helper functions
def row_normalize(mat):
    """
    Row normalize the given matrix mat by its row norm
    """

    # Get the row sum
    inv_r_sum = 1. / torch.norm(mat, dim=1, p=2)
    inv_r_sum[inv_r_sum == torch.inf] = 0.

    # Return normalized matrix
    return mat * inv_r_sum.unsqueeze(1)


# Create the class
class GCNLayer(nn.Module):
    """
    Custom GCN Layer
    """
    def __init__(self, in_channels: int, out_channels: int, device: typing.Optional = torch.device("cpu")):
        super().__init__()

        # Save the device
        self.device = device

        # Create the linear layer with bias always included
        weights = torch.Tensor(in_channels, out_channels).to(self.device)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(out_channels).to(self.device)
        self.bias = nn.Parameter(bias)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weights"""

        # Initialize the weights
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_mat: torch.sparse.Tensor) -> torch.Tensor:
        """Do a forward pass of the network"""

        return torch.mm(torch.sparse.mm(adj_mat, x), self.weights) + self.bias

    def precomputed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass of the network, under the assumption that AX has already been computed"""

        return torch.mm(x, self.weights) + self.bias

# Create the class
class FastGCNv2(nn.Module):
    """
    FastGCN model from Chen, et. al.
    """

    def __init__(self, input_dim: int, hidden_dims: typing.List[int], output_dim: int,
                 csr_mat: sp.csr_matrix, x: torch.Tensor,
                 use_batch_norms: bool = False,
                 dropout: float = 0.0, samp_probs: np.array = None,
                 device: typing.Optional = torch.device("cpu"),
                 dataset_name: str = "", save_path: str = "",
                 seed = 42):

        # Declare super
        super().__init__()

        # Save the device
        self.device = device
        self.use_batch_norms = use_batch_norms

        # Check hidden dims
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            pass

        # Set up the layers
        layer_list = [GCNLayer(in_channels=input_dim, out_channels=hidden_dims[0], device=self.device)]
        batch_norms = [torch.nn.BatchNorm1d(hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layer_list.append(GCNLayer(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], device=self.device))
            batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
        layer_list.append(GCNLayer(in_channels=hidden_dims[-1], out_channels=output_dim, device=self.device))

        # Create a module list
        self.layers = nn.ModuleList(layer_list).to(self.device)
        self.batch_norms = nn.ModuleList(batch_norms).to(self.device)

        # Set activation functions and dropout
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

        # Save the sampler
        self.samp_probs = samp_probs

        # Save the global adjacency matrix for full batch GCN
        if dataset_name == 'ogbn-products':

            # Check for pre-computation matrix
            try:
                self.precompute = torch.load(f"{save_path}/{dataset_name}_precompute.pt")
                self.full_adj = None # IF YOUR GPU HAS ENOUGH MEMORY, CHANGE TO: csr_to_torch_coo(csr_mat).to(self.device)
            except:
                self.full_adj = csr_to_torch_coo(csr_mat)
                self.precompute = torch.sparse.mm(self.full_adj, x)
                torch.save(self.precompute, f"{save_path}/{dataset_name}_precompute.pt")

        else:
            self.full_adj = csr_to_torch_coo(csr_mat).to(self.device)
            self.precompute = torch.sparse.mm(self.full_adj, x)

        self.seed = seed
        # Save the random generator
        self.rng = np.random.default_rng(self.seed)
        N = csr_mat.shape[0]
        self.frequency_count = torch.zeros(N, 1)

    def forward(self, x: torch.Tensor,
                csr_mat: sp.csr_matrix,
                drop: bool = False,
                stochastic: bool = False,
                batch_sizes: typing.List[int] = None,
                possible_training_nodes: list = None) -> tuple:
        """Forward pass of the model"""

        # One way is to perform full pass
        if stochastic is False:

            # No sampling is performed
            init_batch = None

            # Loop through the parameters
            for ind, p in enumerate(self.layers):

                # Check index:
                if ind == 0:

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p.precomputed_forward(self.precompute)) if self.use_batch_norms else p.precomputed_forward(self.precompute)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p.precomputed_forward(self.precompute)) if self.use_batch_norms else p.precomputed_forward(self.precompute)
                        x = self.activation(x)

                elif 0 < ind < (len(self.layers) - 1):

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p(x, self.full_adj)) if self.use_batch_norms else p(x, self.full_adj)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p(x, self.full_adj)) if self.use_batch_norms else p(x, self.full_adj)
                        x = self.activation(x)
                else:

                    # Softmax
                    x = self.final_activation(p(x, self.full_adj))

        # Another is sampling using FastGCN
        else:

            # First get one batch
            init_batch = self.rng.choice(possible_training_nodes,
                                        size=batch_sizes[0],
                                        replace=False)

            # Then compute the subgraphs
            batch_adjs = self.get_subgraphs_concated_sampling(init_batch=init_batch,
                                                     batch_sizes=batch_sizes[1:],
                                                     csr_adj_mat=csr_mat)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Only put the current slice on the machine
                    data = self.precompute[batch_adjs[ind]].to(self.device)

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p.precomputed_forward(data)) if self.use_batch_norms else p.precomputed_forward(data)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p.precomputed_forward(data)) if self.use_batch_norms else p.precomputed_forward(data)
                        x = self.activation(x)

                # Final layer
                elif ind == (len(self.layers) - 1):

                    # Softmax
                    x = self.final_activation(p(x, batch_adjs[ind]))

                # Check index:
                else:

                    # Dropout and activation
                    if drop:
                        x = self.batch_norms[ind](p(x, batch_adjs[ind])) if self.use_batch_norms else p(x, batch_adjs[ind])
                        x = self.drop(self.activation(x))

                    else:
                        x = self.batch_norms[ind](p(x, batch_adjs[ind])) if self.use_batch_norms else p(x, batch_adjs[ind])
                        x = self.activation(x)

        # Return the result
        return x, init_batch

    @torch.no_grad()
    def get_subgraphs_concated_sampling(self, init_batch: typing.List[int], batch_sizes: typing.List[int],
                                       csr_adj_mat: sp.csr_matrix) -> list:
        """Here, we sample from the set of 1-hop neighbors and always include the base nodes"""

        # Create a dummy version of the batch_sizes
        batch_sizes.insert(0, 0)

        # Get the initial batch
        batch = [init_batch]
        adj_out_mats = []
        select_idx = []
        # Get the next layer of nodes
        all_next_nodes = np.unique(csr_adj_mat[batch[0], :].indices)

        # Get only the subset of nodes that are 1-hop away
        new_nodes = np.setdiff1d(all_next_nodes, batch[0])
        old_nodes = batch[0].copy()

        # Loop over the remaining batches
        for i in range(1, len(batch_sizes)):

            # Save denominator to avoid re-computing
            denom = sum(self.samp_probs[new_nodes])

            # Save the probs
            probs = self.samp_probs[new_nodes] / denom

            # Get a batch
            sampled = self.rng.choice(new_nodes, size=min(batch_sizes[i], len(probs)),
                                          replace=False, p=probs)

            batch.append(np.concatenate((sampled, old_nodes)))

            # Create the probability distribution over these nodes
            subgraph_probs = np.concatenate((self.samp_probs[sampled] * len(sampled) / self.samp_probs[sampled].sum(), np.ones(old_nodes.shape)))

            # Save adjmats
            adj_out_mats.append(csr_to_torch_coo(
                csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / subgraph_probs)).to(
                self.device))

            # Get the next layer of nodes
            all_next_nodes = np.unique(csr_adj_mat[sampled, :].indices)

            # Get only the subset of nodes that are 1-hop away
            new_nodes = np.setdiff1d(all_next_nodes, sampled)
            old_nodes = sampled.copy()
            select_idx.extend(old_nodes)
        # Initial layer of the adjacency matrix involves pre-computation,
        # but we already store this, so we only need to slice!
        adj_out_mats.append(batch[-1])
        unique_idx = np.unique(select_idx)
        self.frequency_count[unique_idx] += 1
        return adj_out_mats[::-1]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, csr_mat: sp.csr_matrix) -> tuple:
        """Predict the class for a given set of points"""

        # Forward pass without dropout
        x, _ = self.forward(x, csr_mat, drop=False)

        # Find maximum value for class prediction
        pred = torch.argmax(x, dim=1)

        return pred, x

    @torch.no_grad()
    def sample_predict(self, x: torch.Tensor,
                csr_mat: sp.csr_matrix,
                init_batch: typing.List[int],
                batch_sizes: typing.List[int],
                num_inference_times: int = 1) -> tuple:
        """Perform inference using sampled nodes"""

        # Loop over the different attempts
        final_res = 0

        # Do a stochastic forward pass
        for i in range(num_inference_times):

            # Then compute the subgraphs
            batch_adjs = self.get_subgraphs_concated_sampling(init_batch=init_batch,
                                            batch_sizes=batch_sizes[1:],
                                            csr_adj_mat=csr_mat)

            # Propagate through the network
            for ind, p in enumerate(self.layers):

                # ALWAYS perform precomputation
                if ind == 0:

                    # Activation
                    out = self.batch_norms[ind](p.precomputed_forward(self.precompute[batch_adjs[ind]])) if self.use_batch_norms else p.precomputed_forward(
                        self.precompute[batch_adjs[ind]])
                    out = self.activation(out)

                # Final layer
                elif ind == (len(self.layers) - 1):

                    # Softmax
                    out = self.final_activation(p(out, batch_adjs[ind]))

                # Check index:
                else:

                    # Activation
                    out = self.batch_norms[ind](p(out, batch_adjs[ind])) if self.use_batch_norms else p(out, batch_adjs[ind])
                    out = self.activation(out)

            # Save final results
            final_res += out

        # Scale
        final_res = final_res / num_inference_times

        # Find maximum value for class prediction
        pred = torch.argmax(final_res, dim=1)

        return pred, final_res, init_batch



# Declare colors
cs = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000"
}

# Run main
if __name__=="__main__":

    # # ------------------------------------------------
    # # Parse user arguments
    # parser = argparse.ArgumentParser(description='Test the FastGCN paper method for node classification.')
    #
    # # DATA
    # parser.add_argument('--dataset', type=str, default="Cora", choices=["Cora", "PubMed", "CiteSeer", "Reddit",
    #                                                                     "ogbn-arxiv", "ogbn-products"],
    #                     help='Dataset to use.')
    # parser.add_argument('--norm_feat', type=str, default='false', choices=['true', 'false'],
    #                     help='Normalized features?')
    # parser.add_argument('--batch_norm', type=str, default='false', choices=['true', 'false'],
    #                     help='Use batch normalization?')
    # parser.add_argument('--report', type=int, default=1, help='How often to report accuracies; for bigger data'
    #                                                           'it may be better to take more GD steps first.')
    #
    # # METHOD + ARCHITECTURE
    # parser.add_argument('--fast', type=str, default="true", choices=["true", "false"],
    #                     help='Use FastGCN or regular GCN.')
    # parser.add_argument('--hidden_dim', type=int, default=16, help='Dimension of the hidden layer.')
    # parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers.')
    # parser.add_argument('--init_batch', type=int, default=256, help='Initial batch size.')
    # parser.add_argument('--sample_size', type=int, default=400, help='Sample size size.')
    # parser.add_argument('--scale_factor', type=float, default=1, help='For deeper networks, we need more samples.')
    # parser.add_argument('--samp_dist', type=str, default='importance', choices=['importance', 'uniform'],
    #                     help='Which sampling distribution to use.')
    #
    # # TRAINING
    # parser.add_argument('--epochs', type=int, default=200, help='Total number of updates rounds.')
    # parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate.')
    # parser.add_argument('--early_stop', type=int, default=10, help='Early stopping term.')
    # parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (l2 regularization).')
    # parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate.')
    #
    # # INFERENCE
    # parser.add_argument('--samp_inference', type=str, default='false', choices=['true', 'false'],
    #                     help='Sample during inference phase for testing accuracy?')
    # parser.add_argument('--use_val', type=str, default='true', choices=['true', 'false'],
    #                     help='Use the validation dataset to perform early stopping?')
    # parser.add_argument('--num_samp_inference', type=int, default=1,
    #                     help='Number of times to sample during inference.')
    # parser.add_argument('--inference_init_batch', type=int, default=256,
    #                     help='Initial batch size for inference.')
    # parser.add_argument('--inference_sample_size', type=int, default=400, help='Sample size for inference.')
    #
    # # EXTRAS
    # parser.add_argument('--use_cuda', type=str, default="true", choices=['true', 'false'],
    #                     help='Number of times to sample during inference.')
    # parser.add_argument('--save_results', type=int, default=0, choices=[0, 1],
    #                     help='Save results or not (0 = do NOT save, 1 = save).')
    # parser.add_argument('--seed',       type=int, default=42)
    # parser.add_argument('--label_rate',    type=float, default=1.0,
    #                     help='Fraction of training labels to keep (simulates label scarcity).')
    # parser.add_argument('--patience', type=int, default=1, help='number of epochs to early stopping after achieving best val score.')
    # args = parser.parse_args()
    parser = argparse.ArgumentParser(description="BVSampler")

    # DATA
    parser.add_argument('--dataset', type=str, default="Cora",
                        choices=["Cora", "PubMed", "CiteSeer", "Reddit",
                                 "ogbn-arxiv", "ogbn-products"])
    parser.add_argument('--norm_feat', type=str, default='false',
                        choices=['true', 'false'])
    parser.add_argument('--batch_norm', type=str, default='false',
                        choices=['true', 'false'])
    parser.add_argument('--report', type=int, default=1)

    # MODEL
    parser.add_argument('--fast', type=str, default="true",
                        choices=["true", "false"])
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--init_batch', type=int, default=256)
    parser.add_argument('--sample_size', type=int, default=400)
    parser.add_argument('--scale_factor', type=float, default=1.0)

    # LABEL-SCARCE SETTING
    parser.add_argument('--label_rate', type=float, default=1.0,
                        help='Fraction of training labels to keep (1.0 = all).')
    parser.add_argument('--labeled_ratio', type=float, default=1.0,
                        help='p: fraction of each mini-batch drawn from labeled '
                             'nodes (default 0.80 → 80%% labeled, 20%% unlabeled).')

    # CONTRASTIVE LEARNING
    parser.add_argument('--proj_dim', type=int, default=50,
                        help='Projection head output dimension.')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='InfoNCE temperature τ. Smaller = harder negatives.')
    parser.add_argument('--con_weight', type=float, default=0.5,
                        help='Loss weight λ_c for the unlabeled InfoNCE term. '
                             'Set to 0 to disable contrastive loss.')

    # TRAINING
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)

    # INFERENCE
    parser.add_argument('--samp_inference', type=str, default='false',
                        choices=['true', 'false'])
    parser.add_argument('--use_val', type=str, default='true',
                        choices=['true', 'false'])
    parser.add_argument('--num_samp_inference', type=int, default=1)
    parser.add_argument('--inference_init_batch', type=int, default=256)
    parser.add_argument('--inference_sample_size', type=int, default=400)

    # EXTRAS
    parser.add_argument('--use_cuda', type=str, default="true",
                        choices=['true', 'false'])
    parser.add_argument('--save_results', type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    # ------------------------------------------------
    # Get the device
    user_device = torch.device("cuda:0") if torch.cuda.is_available() and args.use_cuda == 'true' else torch.device("cpu")
    args.fast = True if args.fast == 'true' else False
    args.batch_norm = True if args.batch_norm == 'true' else False
    args.samp_inference = True if args.samp_inference == 'true' else False
    args.use_val = True if args.use_val == 'true' else False
    args.norm_feat = True if args.norm_feat == 'true' else False
    args.early_stop = args.epochs + 1 if args.early_stop <= 0 else args.early_stop

    # Set the architecture
    args.hidden_dim = [args.hidden_dim] * args.num_layers
    score_list = []
    # for rd_seed in [47, 48, 49, 50,51]:
    # for rd_seed in [42, 43, 44, 45, 46]:
    for rd_seed in [43]:
        # set_seeds(args.seed)
        rng = set_seeds(rd_seed)
        # ------------------------------------------------
        # Load the data - ToUndirected ensures that we can scan edge_list[0, :] to get all of the neighbors
        if args.dataset in ['ogbn-arxiv', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=args.dataset, root='../data', transform=T.ToSparseTensor())
            # Extract the data
            data = dataset[0]

            # Gather the variables and responses
            X = data.x.to(user_device)
            y = data.y.flatten().to(user_device)

            # Normalize the features
            X = row_normalize(X) if args.norm_feat else X

            # Get the adjacency matrix - save to file because to symmetric is slow
            try:
                adjmat = sp.load_npz(f"../data/{args.dataset}_adjmat.npz")
            except:
                csr_edge_list = data.adj_t.to_symmetric().to_scipy().tocsr()
                csr_edge_list += sp.identity(X.shape[0], format='csr')
                # Normalize the adjacency matrix
                adjmat = normalize_adj(csr_edge_list)
                sp.save_npz(f"../data/{args.dataset}_adjmat.npz", adjmat)

            # Load the fast-gcn probabilities because this calculation too is slow
            try:
                fast_gcn_probs = np.loadtxt(f"../data/{args.dataset}_probs.txt")
            except:
                fast_gcn_probs = np.asarray(adjmat.multiply(adjmat).sum(1)).flatten()
                np.savetxt(f"../data/{args.dataset}_probs.txt", fast_gcn_probs)

            # Get the training and testing split
            t0 = time.time()
            split_idx = dataset.get_idx_split()
            training_mask = torch.tensor([False] * len(y))
            training_mask[split_idx['train']] = True
            training_mask = training_mask.to(user_device)
            training_indices = split_idx['train'].cpu().numpy()
            testing_indices = split_idx['test'].cpu().numpy()
            validation_indices = split_idx['valid'].cpu().numpy()

            # Standardize data location
            test_mask = torch.tensor([False] * len(y))
            test_mask[testing_indices] = True
            data.test_mask = test_mask

            val_mask = torch.tensor([False] * len(y))
            val_mask[validation_indices] = True
            data.val_mask = val_mask

        else:

            dataset = Reddit(root='../data', transform=T.ToSparseTensor(remove_edge_index=False)) \
                if args.dataset == 'Reddit' \
                else Planetoid(root='../data', name=args.dataset, transform=T.ToSparseTensor(remove_edge_index=False))
            data = dataset[0]

            # Gather the variables and responses
            X = data.x.to(user_device)
            y = data.y.to(user_device)

            # Normalize the features
            X = row_normalize(X) if args.norm_feat else X

            # Create the adjacency matrix
            numpy_edges = data.edge_index.numpy()
            csr_edge_list = sp.csr_matrix((np.ones(data.edge_index.shape[1]), # data
                                           (numpy_edges[0], numpy_edges[1])), # (row, col)
                                          shape=(X.shape[0], X.shape[0])) # size
            csr_edge_list += sp.identity(X.shape[0], format='csr')

            # Calculate adjacency matrix and probabilities
            adjmat = normalize_adj(csr_edge_list)
            fast_gcn_probs = np.asarray(adjmat.multiply(adjmat).sum(1)).flatten()

            # Get the training and testing masks
            training_mask = torch.tensor([True] * len(y))
            training_mask = training_mask * (data.test_mask == False) * (data.val_mask == False)
            training_mask = training_mask.to(user_device)
            training_indices = torch.where(training_mask == True)[0].cpu().numpy()
            testing_indices = torch.where(data.test_mask == True)[0].cpu().numpy()
            validation_indices = torch.where(data.val_mask == True)[0].cpu().numpy()

        if args.label_rate < 1.0:
            if os.path.exists(f"../data/{args.dataset}_{args.label_rate}_label.npz"):
                with np.load(f"../data/{args.dataset}_{args.label_rate}_label.npz", allow_pickle=True) as loaded:
                    try:
                        train_test_data_split = dict(loaded)['arr_0'].item()
                    except:
                        train_test_data_split = dict(loaded)
                    labeled_indices = train_test_data_split['labeled_indices']
                    unlabeled_indices = train_test_data_split['unlabeled_indices']
            else:
                rng_seed = np.random.default_rng(42)
                n_keep = max(1, int(len(training_indices) * args.label_rate))
                keep_idx = rng_seed.choice(len(training_indices), size=n_keep, replace=False)
                labeled_indices = training_indices[keep_idx]
                unlabeled_indices = np.setdiff1d(training_indices, labeled_indices)
                print(f"[SCARCE] Keeping {n_keep}/{len(training_indices)} labeled nodes "
                      f"(label_rate={args.label_rate})")
                train_test_data_split = {'labeled_indices': labeled_indices, 'unlabeled_indices': unlabeled_indices}
                np.savez(f"../data/{args.dataset}_{args.label_rate}_label.npz", train_test_data_split, allow_pickle=True)
            training_indices = labeled_indices
            # data.test_mask = np.concatenate((data.test_mask, unlabeled_indices), axis=0)
            testing_indices = np.concatenate((testing_indices, unlabeled_indices), axis=0)

        # ------------------------------------------------
        # Declare the model and optimizer
        # model = FastGCNv2(input_dim=X.shape[1], hidden_dims=args.hidden_dim, output_dim=max(y).item() + 1, dropout=args.drop,
        #                 csr_mat=adjmat, x=X,
        #                 samp_probs=np.ones((len(y),)) if args.samp_dist == 'uniform' else fast_gcn_probs,
        #                 device=user_device,
        #                 use_batch_norms=args.batch_norm,
        #                 dataset_name=args.dataset,
        #                 save_path='data', seed=args.seed
        #                 )
        norm_cache = HistoricalNormCache(num_nodes=X.shape[0], init_norm=1.0, device=user_device)
        infonce_loss = InfoNCELoss(temperature=args.temperature).to(user_device)
        model = OptISGCN(
            input_dim=X.shape[1],
            hidden_dims=args.hidden_dim,
            output_dim=int(y.max().item()) + 1,
            csr_mat=adjmat,
            x=X,
            norm_cache=norm_cache,
            dropout=args.drop,
            use_batch_norms=args.batch_norm,
            proj_dim=args.proj_dim,
            device=user_device,
            dataset_name=args.dataset,
            save_path='data',
            infoNCELoss=infonce_loss,
            rng=rng,
        ).to(user_device)
        print(f"Your model:\n{model}")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        criteria = nn.NLLLoss(reduction='mean')
        batches = [args.init_batch] + [min(X.shape[0], int(args.sample_size * (1 if i == 0 else args.scale_factor))) for i in range(len(model.layers) - 1)]
        inference_batches = [args.inference_init_batch] + [min(X.shape[0], int(args.inference_sample_size * (1 if i == 0 else args.scale_factor))) for i in range(len(model.layers) - 1)]

        # ------------------------------------------------
        # Save meaningful results
        loss_hist = []
        val_hist = []
        test_acc = []

        # ------------------------------------------------
        # Train the model
        print(f"{'=' * 25} STARTING TRAINING {'=' * 25}")
        print(f"TRAINING INFORMATION:")
        print(f"[DATA] {args.dataset} dataset")
        print(f"[FAST] using FastGCN? {args.fast}")
        print(f"[INF] using sampling for inference? {args.samp_inference}")
        print(f"[FEAT] normalized features? {args.norm_feat}")
        print(f"[DEV] device: {user_device}")
        print(f"[ITERS] performing {args.epochs} Adam updates")
        print(f"[LR] Adam learning rate: {args.lr}")

        if args.fast:
            print(f"[BATCH] batch size: {batches}")

        if args.samp_inference:
            print(f"[INF BATCH] batch size: {inference_batches}")

        # Set the training type
        stochastic = args.fast

        # Perform the for loop over the iterations
        max_acc = 0
        running_time = 0
        total_times = []
        best_val_acc = 0
        valacc_list = []
        patience = args.early_stop
        for i in range(1, args.epochs + 1):

            # Perform training
            t0 = time.time()
            # loss_hist = train(model, optimizer, X, y, adjmat, training_indices, criteria, stochastic, loss_hist,
            #                   batches, training_indices)
            loss_hist = model_train(
                model             = model,
                opt               = optimizer,
                x                 = X,
                y                 = y,
                csr_mat           = adjmat,
                training_mask     = training_indices,
                loss_fn           = criteria,
                stoch             = stochastic,
                losses            = loss_hist,
                batch_list        = batches,
                labeled_indices   = labeled_indices,
                unlabeled_indices = unlabeled_indices,
                labeled_ratio     = args.labeled_ratio,
                rng               = rng,
                # infonce_loss      = infonce_loss,
                con_weight        = args.con_weight,
                device            = user_device,
            )
            t1 = time.time()
            running_time += t1 - t0
            if i > 0:
                total_times.append(t1 - t0)

            # Only report every few iterations
            if i % args.report == 0 and i > 1:

                # Perform testing and validation
                if args.samp_inference:

                    test_acc = sample_test(model, X, y, adjmat, inference_batches, args.num_samp_inference, testing_indices, test_acc)
                    valacc_list = sample_test(model, X, y, adjmat, inference_batches, args.num_samp_inference, validation_indices, valacc_list)
                    # Save validation time if needed
                    if args.use_val:
                        val_hist = sample_validation_test(model, X, y, adjmat, inference_batches, args.num_samp_inference,
                                                          validation_indices, criteria, val_hist)
                # No sampling
                else:
                    test_acc = test(model, X, y, adjmat, testing_indices, test_acc)
                    valacc_list = test(model, X, y, adjmat, validation_indices, valacc_list)
                    # Save validation time if needed
                    if args.use_val:
                        val_hist = validation_test(model, X, y, adjmat, validation_indices, criteria, val_hist)
                val_acc = valacc_list[-1]
                # Check the validation performance ONLY if we are not performing sampled inference
                max_acc = max(test_acc)
                print(f"[{i:04d}] loss={loss_hist[-1]:.4f}  val_acc={val_acc:.2f}%  test_acc={test_acc[-1]:.2f}%  ")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_iter = i
                    patience = 0
                    torch.save(model.state_dict(), "tmp/fastgcn_redditModel.pt")
                    print(f"Found best acc ({val_acc:.5f}) at iteration of {best_iter}. Saving best model...")
                else:
                    patience += 1

        if os.path.exists("tmp/fastgcn_redditModel.pt"):
            model.load_state_dict(
                torch.load("tmp/fastgcn_redditModel.pt", map_location=user_device)
            )
            print(f"Loading the best model at iteration of {best_iter}...")

        final_test_acc = test(model, X, y, adjmat, testing_indices, [])
        # ── Results ───────────────────────────────────────────────────────────
        print(f"\nRESULTS:")
        print(f"[LOSS]  min training loss : {min(loss_hist):.5f}")
        print(f"[ACC]   micro-F1 test : {final_test_acc[-1]:.2f} %")
        print(f"[TIME]  avg per epoch     : {round(sum(total_times) / len(total_times), 4)} s")
        print(f"[TIME]  total training    : {round(running_time, 4)} s")
        print(f"{'=' * 26} ENDING TRAINING {'=' * 26}\n")
        score_list.append(final_test_acc)

        from collections import Counter

        # Count frequencies
        freq = Counter(model.frequency_count.cpu().numpy().reshape(-1, ))

        # Separate keys and values
        x = list(freq.keys())
        y = list(freq.values())

        # Plot
        plt.figure()
        plt.bar(x, y)
        plt.xlabel("Data Points")
        plt.ylabel("Frequency")
        plt.title("Frequency of Data Points")
        plt.show()

        # ── Plots ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[0].plot(loss_hist, color=cs["blue"], lw=2.5, label='train loss')
        if val_hist:
            axes[0].plot(np.linspace(0, len(loss_hist) - 1, len(val_hist)),
                         val_hist, color=cs["red"], lw=2.5, label='val loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss (log scale)')
        axes[0].set_yscale('log')
        axes[0].legend(frameon=True)
        axes[0].set_title('Training & Validation Loss')

        axes[1].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].plot(test_acc, color=cs["orange"], lw=2.5, label='OptIS-GCN')
        axes[1].set_xlabel('Evaluation step')
        axes[1].set_ylabel('Test accuracy (micro-F1 %)')
        axes[1].legend(frameon=True)
        axes[1].set_title('Test Accuracy')

        plt.tight_layout()
        plt.show()
        if args.save_results:
            plt.savefig(f"results/{args.dataset}_optis_gcn.png",
                        transparent=True, bbox_inches='tight', pad_inches=0)
    print(score_list)
    print('Final average score:', np.mean(score_list), ' with std: ', np.std(score_list))
    np.save('val_f1_list_{}.npy'.format(args.dataset), valacc_list)
    np.save('test_f1_list_{}.npy'.format(args.dataset), test_acc)


