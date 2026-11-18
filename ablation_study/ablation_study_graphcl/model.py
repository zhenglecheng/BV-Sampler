#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import typing
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F


class HistoricalNormCache:
    """
    Per-node cache of ‖h_u‖ with staleness-aware sampling bonus.

        norm_cache[u]  – most recently observed ‖h_u‖
        last_seen[u]   – iteration when u was last visited

    Staleness weight:  w(u) = 1 + log(1 + T_u),   T_u = current_iter − last_seen[u]

    Nodes not yet visited receive T = current_iter + 1 (slightly larger bonus),
    ensuring cold-start nodes are not perpetually under-sampled.

    GPU change
    ----------
    norm_cache and last_seen are now torch.Tensors on `device` instead of
    NumPy arrays.  This eliminates the .detach().cpu() call in update() and
    keeps all indexing / arithmetic on the same device as the model.
    """

    def __init__(self, num_nodes: int, init_norm: float = 1.0,
                 device=torch.device("cpu")):
        self.device = device
        self.norm_cache   = torch.full((num_nodes,), init_norm,
                                       dtype=torch.float64, device=device)
        self.last_seen    = torch.full((num_nodes,), -1,
                                       dtype=torch.int64,  device=device)
        self.current_iter = 0

    # ------------------------------------------------------------------
    def _to_idx(self, node_ids) -> torch.Tensor:
        """Convert node_ids (ndarray or Tensor) to a GPU LongTensor."""
        if isinstance(node_ids, torch.Tensor):
            return node_ids.to(self.device, dtype=torch.long)
        return torch.as_tensor(node_ids, dtype=torch.long, device=self.device)

    def update(self, node_ids, h: torch.Tensor, ema_alpha: float = 0.25) -> None:
        idx = self._to_idx(node_ids)
        norms = h.detach().norm(dim=1).to(dtype=self.norm_cache.dtype, device=self.device)
        self.norm_cache[idx] = (1.0 - ema_alpha) * self.norm_cache[idx] + ema_alpha * norms
        self.last_seen[idx] = self.current_iter

    def get_norms(self, node_ids) -> torch.Tensor:
        """Returns a GPU float64 tensor of cached norms."""
        return self.norm_cache[self._to_idx(node_ids)]

    def staleness_weights(self, node_ids) -> torch.Tensor:
        """Returns a GPU float64 tensor of staleness weights."""
        idx  = self._to_idx(node_ids)
        last = self.last_seen[idx].to(torch.float64)
        T    = torch.where(
            last < 0,
            torch.tensor(float(self.current_iter + 1),
                         dtype=torch.float64, device=self.device),
            torch.tensor(float(self.current_iter),
                         dtype=torch.float64, device=self.device) - last,
        )
        return 1.0 + torch.log1p(T)

    def step(self) -> None:
        self.current_iter += 1


class ProjectionHead(nn.Module):
    """
    Two-layer MLP: h → BN → ReLU → Linear → L2-normalise → z.

    Applied to target-aligned encoder representations before the GraphCL loss.
    Keeping the contrastive objective in projection space (not h) lets the
    encoder preserve task-relevant structure for the classification head.
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=1)


class GraphCLLoss(nn.Module):
    """Symmetric node-level NT-Xent; both views contain the same target IDs."""
    def __init__(self, temperature=0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.tau = temperature

    def forward(self, z1, z2):
        if z1.shape != z2.shape or z1.ndim != 2 or len(z1) < 2:
            raise ValueError("GraphCL requires two aligned (N,D) views with N >= 2")
        z = F.normalize(torch.cat([z1, z2]), dim=1)
        logits = z @ z.t() / self.tau
        logits = logits.masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -torch.inf)
        targets = (torch.arange(len(z), device=z.device) + len(z1)) % len(z)
        return F.cross_entropy(logits, targets)


# Compatibility for imports in existing experiment scripts.
InfoNCELoss = GraphCLLoss


class GCNLayer(nn.Module):
    """
    Custom GCN Layer
    """
    def __init__(self, in_channels: int, out_channels: int, device: typing.Optional = torch.device("cpu")):
        super().__init__()
        self.device = device
        weights = torch.Tensor(in_channels, out_channels).to(self.device)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(out_channels).to(self.device)
        self.bias = nn.Parameter(bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weights"""
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_mat: torch.sparse.Tensor) -> torch.Tensor:
        """Do a forward pass of the network"""
        return torch.mm(torch.sparse.mm(adj_mat, x), self.weights) + self.bias

    def precomputed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do a forward pass of the network, under the assumption that AX has already been computed"""
        return torch.mm(x, self.weights) + self.bias



def csr_to_torch_coo(adj):
    """
    Convert csr sparse matrix to coo sparse tensor
    """
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)
    return torch.sparse_coo_tensor(indices=np.array([adj.row, adj.col]), values=adj.data, size=adj.shape, dtype=torch.float32)



# Create the class
class OptISGCN(nn.Module):
    """
    FastGCN model from Chen, et. al.
    """

    def __init__(self, input_dim: int, hidden_dims: typing.List[int], output_dim: int,
                 csr_mat: sp.csr_matrix, x: torch.Tensor,
                 use_batch_norms: bool = False,
                 dropout: float = 0.0, samp_probs: np.array = None,
                 device: typing.Optional = torch.device("cpu"),
                 dataset_name: str = "", save_path: str = "",
                 seed=42,
                 norm_cache: HistoricalNormCache = None,
                 infoNCELoss: InfoNCELoss = None,
                 rng: np.random.Generator = None,
                 proj_dim: int = 50, feature_mask_rate: float = 0.2,
                 ):
        super().__init__()
        self.device = device
        self.use_batch_norms = use_batch_norms
        self.norm_cache = norm_cache
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        else:
            pass

        # Set up the layers
        layer_list = [GCNLayer(in_channels=input_dim, out_channels=hidden_dims[0], device=self.device)]
        batch_norms = [torch.nn.BatchNorm1d(hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layer_list.append(
                GCNLayer(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], device=self.device))
            batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))
        layer_list.append(GCNLayer(in_channels=hidden_dims[-1], out_channels=output_dim, device=self.device))

        # Create a module list
        self.layers = nn.ModuleList(layer_list).to(self.device)
        self.batch_norms = nn.ModuleList(batch_norms).to(self.device)

        # Set activation functions and dropout
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

        # Save the sampler
        self.samp_probs = torch.tensor(samp_probs).to(self.device)

        if not 0 <= feature_mask_rate < 1:
            raise ValueError("feature_mask_rate must be in [0, 1)")
        self.feature_mask_rate = feature_mask_rate
        self.proj_head = ProjectionHead(hidden_dims[-1], proj_dim).to(self.device)
        # Keep raw graph storage, but never propagate over all nodes at once.
        self.csr_mat = csr_mat.tocsr()
        self.raw_features = x.detach().cpu()
        self.num_nodes = x.shape[0]

        self.seed = seed
        # Save the random generator
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=device)
        self.torch_rng.manual_seed(seed)
        N = csr_mat.shape[0]
        self.frequency_count = torch.zeros(N, 1)
        self.infoNCELoss = infoNCELoss

    def forward(self, csr_mat: sp.csr_matrix,
                drop: bool = False,
                stochastic: bool = False,
                batch_sizes: typing.List[int] = None,
                init_batch: np.ndarray = None,
                contrastive_batch: np.ndarray = None
                ) -> tuple:
        """Forward pass of the model"""
        if not stochastic:
            raise ValueError("This ablation requires sampled mini-batch propagation")
        else:
            if init_batch is None:
                raise ValueError("init_batch is required")
            if contrastive_batch is not None:
                raise ValueError("Use graphcl_views on the same batch; independent contrastive sampling is disabled")
            batch_adjs, batch_alpha = self._get_biased_subgraphs(
                init_batch=init_batch, batch_sizes=list(batch_sizes[1:]), csr_adj_mat=csr_mat)
            nce_subgraph = (batch_adjs, batch_alpha)
            h = None
            contra_h = None

            for ind, layer in enumerate(self.layers):
                if ind == 0:
                    data = self.first_aggregate(batch_adjs[0])
                    data = data * batch_alpha[0].unsqueeze(1).float()
                    h = layer.precomputed_forward(data)
                    h = self.batch_norms[ind](h) if self.use_batch_norms else h
                    # cache the norm before dropout.
                    self.norm_cache.update(batch_adjs[0], h)
                    h = self.drop(self.activation(h)) if drop else self.activation(h)
                elif ind == len(self.layers) - 1:
                    h = self.final_activation(layer(h, batch_adjs[ind]))
                else:
                    h = layer(h, batch_adjs[ind])
                    h = self.batch_norms[ind](h) if self.use_batch_norms else h
                    h = self.drop(self.activation(h)) if drop else self.activation(h)
                if ind == len(self.layers) - 2:
                    contra_h = h
            self.norm_cache.step()
        return h, contra_h, init_batch, nce_subgraph

    @torch.no_grad()
    def get_subgraphs_concated_sampling(self, init_batch: typing.List[int], batch_sizes: typing.List[int],
                                        csr_adj_mat: sp.csr_matrix, random_rng=False) -> list:
        """Sample from the set of 1-hop neighbors and always include the base nodes.

        Torch-first version: probability computation, normalization, sampling, and
        the alpha/total_scale vector all stay on `self.device`. We only convert back
        to numpy where scipy.sparse demands it (CSR row/col indexing and `.multiply`).
        """
        device = self.device
        batch_sizes = [0] + list(batch_sizes)
        init_batch_np = np.asarray(init_batch)
        batch = [init_batch_np]
        adj_out_mats = []
        select_idx_list = []  # accumulate arrays, concat once at the end
        alpha_out = []

        # First neighborhood (scipy → numpy)
        all_next_nodes = np.unique(csr_adj_mat[batch[0], :].indices)
        new_nodes = np.setdiff1d(all_next_nodes, batch[0])
        old_nodes = batch[0].copy()

        for i in range(1, len(batch_sizes)):
            k = batch_sizes[i]

            # --- probabilities in torch ---
            # samp_probs is a torch tensor; numpy-array indexing into it is supported.
            h_norms = self.norm_cache.get_norms(new_nodes)  # torch tensor
            samp_probs_new = self.samp_probs[new_nodes]  # torch tensor
            new_prob = torch.clamp(samp_probs_new * h_norms, min=1e-12)
            probs = new_prob / new_prob.sum()

            # --- sampling in torch (multinomial, without replacement) ---
            n_sample = min(k, probs.numel())
            if n_sample == 0:
                local_idx = torch.empty(0, dtype=torch.long, device=device)
            elif random_rng:
                torch_rng = torch.Generator(device=self.device)
                local_idx = torch.multinomial(probs.float(), generator=torch_rng, num_samples=n_sample, replacement=False)
            else:
                local_idx = torch.multinomial(probs.float(), generator=self.torch_rng, num_samples=n_sample, replacement=False)

            # Map local positions back to global node IDs (numpy for scipy ops).
            local_idx_np = local_idx.detach().cpu().numpy()
            sampled = new_nodes[local_idx_np]

            # Build the next batch slice (numpy, for scipy indexing).
            batch.append(np.concatenate((sampled, old_nodes)))

            # --- alpha / total_scale in torch ---
            samp_probs_sampled = samp_probs_new[local_idx]  # reuse, no re-gather
            kp = k * samp_probs_sampled
            alphas = kp / (1.0 + kp)
            scale_sampled = samp_probs_sampled * sampled.shape[0] / (alphas * samp_probs_sampled.sum())
            total_scale = torch.cat([
                scale_sampled,
                torch.ones(old_nodes.shape[0], dtype=scale_sampled.dtype, device=device),
            ])

            # scipy.sparse.multiply needs a numpy array.
            total_scale_np = total_scale.detach().cpu().numpy()
            adj_out_mats.append(
                csr_to_torch_coo(
                    csr_adj_mat[batch[i - 1], :][:, batch[i]].multiply(1. / total_scale_np)
                ).to(device)
            )

            # Next hop (scipy → numpy)
            all_next_nodes = np.unique(csr_adj_mat[sampled, :].indices)
            new_nodes = np.setdiff1d(all_next_nodes, sampled)
            old_nodes = sampled.copy() if len(sampled) else batch[-1].copy()

            select_idx_list.append(old_nodes)
            alpha_out.append(total_scale)  # kept as a torch tensor for downstream use

        # Last layer is precomputed elsewhere — just slot in batch[-1].
        adj_out_mats.append(batch[-1])

        # Bookkeeping
        if select_idx_list:
            unique_idx = np.unique(np.concatenate(select_idx_list))
        else:
            unique_idx = np.empty(0, dtype=np.int64)

        bottom_alpha = alpha_out[-1] if alpha_out else torch.ones(len(batch[-1]), device=device)
        alpha_out.append(bottom_alpha)
        self.frequency_count[unique_idx] += 1

        return adj_out_mats[::-1], alpha_out[::-1]

    @torch.no_grad()
    def sample_predict(self, csr_mat: sp.csr_matrix,
                       init_batch: typing.List[int],
                       batch_sizes: typing.List[int],
                       num_inference_times: int = 1) -> tuple:
        """Perform inference using sampled nodes"""

        # Loop over the different attempts
        final_res = 0
        for i in range(num_inference_times):
            # Then compute the subgraphs
            batch_adjs, batch_alpha = self.get_subgraphs_concated_sampling(init_batch=init_batch, batch_sizes=batch_sizes[1:], csr_adj_mat=csr_mat)
            # Propagate through the network
            for ind, p in enumerate(self.layers):
                # ALWAYS perform precomputation
                if ind == 0:
                    data = self.first_aggregate(batch_adjs[ind])
                    # scale = torch.tensor(batch_alpha[ind], dtype=torch.float32, device=self.device)
                    # data = data * scale.unsqueeze(1)
                    data = data * batch_alpha[0].unsqueeze(1).float()
                    out = p.precomputed_forward(data)
                    if self.use_batch_norms:
                        out = self.batch_norms[ind](out)
                    out = self.activation(out)
                # Final layer
                elif ind == (len(self.layers) - 1):
                    out = self.final_activation(p(out, batch_adjs[ind]))
                else:
                    out = self.batch_norms[ind](p(out, batch_adjs[ind])) if self.use_batch_norms else p(out, batch_adjs[ind])
                    out = self.activation(out)
            final_res += out
        # Scale
        final_res = final_res / num_inference_times
        # Find maximum value for class prediction
        pred = torch.argmax(final_res, dim=1)
        return pred, final_res, init_batch

    def predict(self, x: torch.Tensor, csr_mat: sp.csr_matrix) -> tuple:
        out, _, _, _ = self.forward(csr_mat, drop=False, stochastic=False)
        return torch.argmax(out, dim=1), out

    @torch.no_grad()
    def first_aggregate(self, node_ids):
        """Compute A[batch_rows, neighbors] X[neighbors], without a global AX."""
        rows = self.csr_mat[np.asarray(node_ids, dtype=np.int64)]
        neighbors = np.unique(rows.indices)
        local_adj = csr_to_torch_coo(rows[:, neighbors]).to(self.device)
        features = self.raw_features[torch.as_tensor(neighbors)].to(self.device)
        return torch.sparse.mm(local_adj, features)

    def _masked_target_embeddings(self, batch_adjs, alpha_scales, first_features, target_positions, mask):
        # A(X*M) = (AX)*M exactly for a feature mask shared across nodes.
        data = first_features * mask.unsqueeze(0)
        data = data * alpha_scales[0].to(device=self.device, dtype=data.dtype).unsqueeze(1)
        h = self.layers[0].precomputed_forward(data)
        h = self.batch_norms[0](h) if self.use_batch_norms else h
        h = self.activation(h)
        for ind in range(1, len(self.layers) - 1):
            h = self.layers[ind](h, batch_adjs[ind])
            h = self.batch_norms[ind](h) if self.use_batch_norms else h
            h = self.activation(h)
        # Penultimate nodes are NOT necessarily target nodes. Aggregate using
        # the final sampled adjacency before selecting unlabeled targets.
        h = torch.sparse.mm(batch_adjs[-1], h)
        return self.proj_head(h[target_positions])

    def graphcl_views(self, batch_adjs, alpha_scales, target_positions):
        """Two feature-only views sharing every sampled node, edge and weight."""
        first_features = self.first_aggregate(batch_adjs[0])
        positions = torch.as_tensor(target_positions, device=self.device, dtype=torch.long)
        views = []
        for _ in range(2):
            mask = (torch.rand(first_features.shape[1], device=self.device,
                               generator=self.torch_rng) >= self.feature_mask_rate).to(first_features.dtype)
            views.append(self._masked_target_embeddings(
                batch_adjs, alpha_scales, first_features, positions, mask))
        return tuple(views)

    @torch.no_grad()
    def _get_biased_subgraphs(self, init_batch: np.ndarray,
                               batch_sizes: typing.List[int],
                               csr_adj_mat: sp.csr_matrix,
                               random_rng: bool = False,
                               contrastive_batch: np.ndarray = None) -> tuple:
        adj_out_gcn, alpha_out_gcn = self.get_subgraphs_concated_sampling(init_batch, batch_sizes, csr_adj_mat, random_rng=random_rng)
        return adj_out_gcn, alpha_out_gcn

    @torch.no_grad()
    def init_norm_cache(self, batch_size: int = 1024) -> None:
        N      = self.num_nodes
        layer0 = self.layers[0]
        print(f"[CACHE] Initialising norm cache for {N} nodes …")
        for start in range(0, N, batch_size):
            idx  = np.arange(start, min(start + batch_size, N))
            data = self.first_aggregate(idx)
            h    = self.activation(layer0.precomputed_forward(data))
            self.norm_cache.update(idx, h)
        self.norm_cache.step()
        print("[CACHE] Done.")

