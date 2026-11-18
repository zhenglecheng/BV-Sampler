#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import typing
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F


class HistoricalNormCache:
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


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N, device = z1.size(0), z1.device
        sim  = torch.exp(torch.mm(z1, z2.t()) / self.tau)
        loss = torch.mean(torch.log(sim.sum(0) / (sim.diag() * N)))
        return loss

    # def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    #     """
    #     z1, z2 : (N, D)  L2-normalised projections for the same N nodes.
    #     Returns scalar NT-Xent loss.
    #     """
    #
    #
    #     z   = torch.cat([z1, z2], dim=0)                          # (2N, D)
    #     N, device = z.size(0), z.device
    #     # Z_norm = torch.norm(z, dim=1, keepdim=True)
    #     # sim = torch.mm(z, z.t()) / self.tau                       # (2N, 2N)
    #     sim = torch.mm(z, z.t()) /self.tau                    # (2N, 2N)
    #     sim.masked_fill_(torch.eye(N, dtype=torch.bool, device=device), -1e9)
    #
    #     # view-1 row i  → positive at column i+N
    #     # view-2 row i+N → positive at column i
    #     labels = torch.cat([
    #         torch.arange(N, N, device=device),
    #         torch.arange(0, N,     device=device),
    #     ])
    #     return F.cross_entropy(sim, labels)/(N*N)


class GCNLayer(nn.Module):
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
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)
    return torch.sparse_coo_tensor(
        indices=np.array([adj.row, adj.col]),
        values=adj.data,
        size=adj.shape,
        dtype=torch.float32,
    ).coalesce()



# Create the class
class BVSampler(nn.Module):
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
                 proj_dim: int = 128,
                 alpha_mode: str = "optimal",
                 sampling_replacement: bool = False,
                 ):
        super().__init__()
        self.device = device
        self.use_batch_norms = use_batch_norms
        self.norm_cache = norm_cache
        if alpha_mode not in {"optimal", "one"}:
            raise ValueError("alpha_mode must be either 'optimal' or 'one'.")
        self.alpha_mode = alpha_mode
        self.sampling_replacement = bool(sampling_replacement)
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

        # Save the global adjacency matrix for full batch GCN
        if dataset_name == 'ogbn-products':
            # Check for pre-computation matrix
            try:
                self.precompute = torch.load(f"{save_path}/{dataset_name}_precompute.pt")
                self.full_adj = None
            except:
                self.full_adj = csr_to_torch_coo(csr_mat)
                self.precompute = torch.sparse.mm(self.full_adj, x)
                torch.save(self.precompute, f"{save_path}/{dataset_name}_precompute.pt")
        else:
            self.full_adj = csr_to_torch_coo(csr_mat).to(self.device)
            self.precompute = torch.sparse.mm(self.full_adj, x)

        self.seed = seed
        # Save the random generator
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device=device)
        self.torch_rng.manual_seed(seed)
        N = csr_mat.shape[0]
        self.frequency_count = torch.zeros(N, 1)
        self.infoNCELoss = infoNCELoss
        self.proj_head = ProjectionHead(hidden_dims[-1], proj_dim=proj_dim).to(self.device)

    def reset_sampling_seed(self, seed: int) -> None:
        """Reset the sampler RNG so Monte Carlo diagnostics are reproducible."""
        self.torch_rng.manual_seed(int(seed))

    @torch.no_grad()
    def full_final_aggregation(self, csr_mat: sp.csr_matrix,
                               init_batch: typing.Sequence[int]) -> torch.Tensor:
        """Compute the exact final-layer neighborhood aggregation for fixed nodes.

        This returns A H at the input to the final classifier, before the final
        weight matrix, bias, and LogSoftmax. It is therefore a closer empirical
        counterpart of the aggregation estimator analyzed in the paper than the
        final class probabilities.
        """
        self.eval()
        node_ids = np.asarray(init_batch, dtype=np.int64)

        # Exact hidden states for every node. The first adjacency multiplication
        # is already cached in self.precompute = A X.
        h = self.layers[0].precomputed_forward(self.precompute)
        h = self.batch_norms[0](h) if self.use_batch_norms else h
        h = self.activation(h)

        for ind in range(1, len(self.layers) - 1):
            if self.full_adj is None:
                raise RuntimeError(
                    "full_final_aggregation requires self.full_adj for models "
                    "with more than one hidden layer.")
            h = self.layers[ind](h, self.full_adj)
            h = self.batch_norms[ind](h) if self.use_batch_norms else h
            h = self.activation(h)

        # Only materialize the final adjacency rows needed for evaluation.
        target_adj = csr_to_torch_coo(csr_mat[node_ids, :]).to(self.device)
        return torch.sparse.mm(target_adj, h)

    @torch.no_grad()
    def sampled_final_aggregation(
        self,
        csr_mat: sp.csr_matrix,
        init_batch: typing.Sequence[int],
        batch_sizes: typing.Sequence[int],
        alpha_mode: typing.Optional[str] = None,
        replacement: typing.Optional[bool] = None,
        return_sampling_stats: bool = False,
    ):
        """Draw one sampled pass and return the final aggregation ``A_hat H_hat``.

        Parameters
        ----------
        alpha_mode:
            Optional evaluation-time override. This makes it possible to compare
            ``alpha=1`` and ``alpha=kp/(1+kp)`` using the same frozen model.
        replacement:
            Set to ``True`` for the i.i.d. sampling design assumed by the theory.
            Resetting ``self.torch_rng`` before two calls gives paired draws.
        return_sampling_stats:
            When true, also return per-hop alpha values and sampling metadata.

        The first hidden representation is computed from ``self.precompute = AX``.
        It must *not* be multiplied by the importance scale: the sampled sparse
        adjacency matrices already contain the factor ``alpha/(k p)``.
        """
        self.eval()
        node_ids = np.asarray(init_batch, dtype=np.int64)
        local_batch_sizes = list(batch_sizes)

        sampled = self.get_subgraphs_concated_sampling(
            init_batch=node_ids,
            batch_sizes=local_batch_sizes[1:],
            csr_adj_mat=csr_mat,
            alpha_mode=alpha_mode,
            replacement=replacement,
            return_stats=return_sampling_stats,
        )
        if return_sampling_stats:
            batch_adjs, _, sampling_stats = sampled
        else:
            batch_adjs, _ = sampled
            sampling_stats = None

        data = self.precompute[batch_adjs[0]].to(self.device)
        h = self.layers[0].precomputed_forward(data)
        h = self.batch_norms[0](h) if self.use_batch_norms else h
        h = self.activation(h)

        for ind in range(1, len(self.layers) - 1):
            h = self.layers[ind](h, batch_adjs[ind])
            h = self.batch_norms[ind](h) if self.use_batch_norms else h
            h = self.activation(h)

        output = torch.sparse.mm(batch_adjs[-1], h)
        if return_sampling_stats:
            return output, sampling_stats
        return output

    def forward(self, csr_mat: sp.csr_matrix,
                drop: bool = False,
                stochastic: bool = False,
                batch_sizes: typing.List[int] = None,
                init_batch: np.ndarray = None,
                contrastive_batch: np.ndarray = None
                ) -> tuple:
        """Forward pass of the model"""
        # One way is to perform full pass
        if stochastic is False:
            for ind, p in enumerate(self.layers):
                if ind == 0:
                    if drop:
                        x = self.batch_norms[ind](p.precomputed_forward(self.precompute)) if self.use_batch_norms else p.precomputed_forward(self.precompute)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p.precomputed_forward(self.precompute)) if self.use_batch_norms else p.precomputed_forward(self.precompute)
                        x = self.activation(x)
                elif 0 < ind < (len(self.layers) - 1):
                    if drop:
                        x = self.batch_norms[ind](p(x, self.full_adj)) if self.use_batch_norms else p(x, self.full_adj)
                        x = self.drop(self.activation(x))
                    else:
                        x = self.batch_norms[ind](p(x, self.full_adj)) if self.use_batch_norms else p(x, self.full_adj)
                        x = self.activation(x)
                else:
                    x = self.final_activation(p(x, self.full_adj))
            return x, None, None, None
        # BVSampler
        else:
            assert init_batch is not None, "init_batch must be provided when stochastic=True"
            nce_subgraph = None
            do_contrastive = (contrastive_batch is not None and len(contrastive_batch) > 1)
            if do_contrastive:
                (batch_adjs, batch_alpha), (adj_out_nce, alpha_out_nce) = \
                    self._get_biased_subgraphs(
                        init_batch        = init_batch,
                        batch_sizes       = batch_sizes[1:],
                        csr_adj_mat       = csr_mat,
                        contrastive_batch = contrastive_batch,
                    )
                nce_subgraph = (adj_out_nce, alpha_out_nce)
            else:
                batch_adjs, batch_alpha = self._get_biased_subgraphs(
                    init_batch  = init_batch,
                    batch_sizes = batch_sizes[1:],
                    csr_adj_mat = csr_mat,
                )
            h = None
            contra_h = None

            for ind, layer in enumerate(self.layers):
                if ind == 0:
                    data = self.precompute[batch_adjs[0]].to(self.device)
                    h = layer.precomputed_forward(data)
                    h = self.batch_norms[ind](h) if self.use_batch_norms else h
                    # cache the norm before dropout.
                    self.norm_cache.update(batch_adjs[0], h)
                    h = self.drop(self.activation(h)) if drop else self.activation(h)
                elif ind == len(self.layers) - 1:
                    if do_contrastive:
                        # The final adjacency rows follow the mixed target order.
                        contra_h = torch.sparse.mm(batch_adjs[ind], h)
                    h = self.final_activation(layer(h, batch_adjs[ind]))
                else:
                    h = layer(h, batch_adjs[ind])
                    h = self.batch_norms[ind](h) if self.use_batch_norms else h
                    h = self.drop(self.activation(h)) if drop else self.activation(h)
            self.norm_cache.step()
        return h, contra_h, init_batch, nce_subgraph

    @torch.no_grad()
    def get_subgraphs_concated_sampling(
        self,
        init_batch: typing.Sequence[int],
        batch_sizes: typing.Sequence[int],
        csr_adj_mat: sp.csr_matrix,
        random_rng: bool = False,
        alpha_mode: typing.Optional[str] = None,
        replacement: typing.Optional[bool] = None,
        return_stats: bool = False,
    ):
        """Construct sampled layer-wise subgraphs.

        The sampling probability used for drawing a node is also used in the
        importance correction. For a sampled occurrence ``u``, the sparse
        adjacency column is multiplied by

            alpha(u) / (k * p(u)).

        The current frontier is included deterministically so its self/skip
        contributions are exact. Neighbor expansion is performed from the *full*
        current frontier, rather than only from the most recently sampled nodes.

        Notes
        -----
        ``replacement=True`` matches the i.i.d. estimator in the theory. Weighted
        sampling without replacement uses the same first-order correction but is
        not exactly unbiased unless every candidate is selected.
        """
        del random_rng  # sequential calls to self.torch_rng already give independent views
        device = self.device
        effective_alpha_mode = alpha_mode or self.alpha_mode
        if effective_alpha_mode not in {"optimal", "one"}:
            raise ValueError("alpha_mode must be either 'optimal' or 'one'.")
        use_replacement = self.sampling_replacement if replacement is None else bool(replacement)

        sizes = [0] + [int(v) for v in batch_sizes]
        batch = [np.asarray(init_batch, dtype=np.int64)]
        adj_out_mats = []
        alpha_out = []
        selected_global_ids = []
        sampling_stats = []

        for hop in range(1, len(sizes)):
            k = sizes[hop]
            current_rows = np.asarray(batch[hop - 1], dtype=np.int64)
            fixed_nodes = np.unique(current_rows)

            # Every row whose representation is needed contributes to the next
            # frontier. The old implementation expanded only from newly sampled
            # nodes, which omitted neighbors of deterministically retained nodes.
            all_next_nodes = np.unique(csr_adj_mat[current_rows, :].indices)
            new_nodes = np.setdiff1d(all_next_nodes, fixed_nodes, assume_unique=False)

            alpha_values = torch.empty(0, dtype=torch.float64, device=device)
            selected_probs = torch.empty(0, dtype=torch.float64, device=device)
            sampled_nodes = np.empty(0, dtype=np.int64)
            exact_enumeration = False

            if k > 0 and new_nodes.size > 0:
                new_idx = torch.as_tensor(new_nodes, dtype=torch.long, device=device)
                h_norms = self.norm_cache.get_norms(new_nodes).to(torch.float64)
                base_probs = self.samp_probs[new_idx].to(torch.float64)
                unnormalized = torch.clamp(base_probs * h_norms, min=1e-12)
                probs = unnormalized / unnormalized.sum()

                if use_replacement:
                    n_draws = k
                else:
                    n_draws = min(k, int(new_nodes.size))

                if (not use_replacement) and n_draws == new_nodes.size:
                    # There is no sampling randomness when all candidates are
                    # enumerated, so both the unbiased and MSE-optimal estimator
                    # reduce to the exact aggregation (alpha=1, scale=1).
                    local_idx = torch.arange(new_nodes.size, device=device)
                    sampled_nodes = new_nodes.copy()
                    selected_probs = probs
                    alpha_values = torch.ones_like(selected_probs)
                    scale_sampled = torch.ones_like(selected_probs)
                    exact_enumeration = True
                else:
                    local_idx = torch.multinomial(
                        probs.float(),
                        generator=self.torch_rng,
                        num_samples=n_draws,
                        replacement=use_replacement,
                    )
                    sampled_nodes = new_nodes[local_idx.detach().cpu().numpy()]
                    selected_probs = probs[local_idx]
                    kp = float(n_draws) * selected_probs
                    if effective_alpha_mode == "one":
                        alpha_values = torch.ones_like(kp)
                    else:
                        alpha_values = kp / (1.0 + kp)
                    scale_sampled = kp / torch.clamp(alpha_values, min=1e-12)
            else:
                n_draws = 0
                scale_sampled = torch.empty(0, dtype=torch.float64, device=device)

            # Sampled occurrences may repeat under i.i.d. sampling; repeated
            # columns preserve their multiplicity in the Monte Carlo estimator.
            next_nodes = np.concatenate((sampled_nodes, fixed_nodes))
            total_scale = torch.cat([
                scale_sampled,
                torch.ones(fixed_nodes.size, dtype=torch.float64, device=device),
            ])
            batch.append(next_nodes)

            total_scale_np = total_scale.detach().cpu().numpy()
            sampled_adj = csr_adj_mat[current_rows, :][:, next_nodes]
            sampled_adj = sampled_adj.multiply(1.0 / total_scale_np)
            adj_out_mats.append(csr_to_torch_coo(sampled_adj).to(device))
            alpha_out.append(total_scale)

            if sampled_nodes.size:
                selected_global_ids.append(np.unique(sampled_nodes))

            sampling_stats.append({
                "hop": hop,
                "num_rows": int(current_rows.size),
                "num_fixed": int(fixed_nodes.size),
                "num_candidates": int(new_nodes.size),
                "num_draws": int(n_draws),
                "replacement": bool(use_replacement),
                "exact_enumeration": bool(exact_enumeration),
                "alpha_values": alpha_values.detach().cpu(),
                "selected_probs": selected_probs.detach().cpu(),
            })

        # The first item consumed by the network is the node-index array for AX.
        adj_out_mats.append(batch[-1])
        alpha_out.append(torch.ones(len(batch[-1]), dtype=torch.float64, device=device))

        if selected_global_ids:
            unique_idx = np.unique(np.concatenate(selected_global_ids))
            self.frequency_count[unique_idx] += 1

        result = (adj_out_mats[::-1], alpha_out[::-1])
        if return_stats:
            return result[0], result[1], sampling_stats
        return result

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
                    data = self.precompute[batch_adjs[ind]].to(self.device)
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

    def forward_embed_presampled(self, batch_adjs: list, alpha_scales: list,
                                  drop: bool = True) -> torch.Tensor:
        del alpha_scales  # correction is already encoded in batch_adjs
        data  = self.precompute[batch_adjs[0]].to(self.device)
        h     = self.layers[0].precomputed_forward(data)
        h     = self.batch_norms[0](h) if self.use_batch_norms else h
        h     = self.drop(self.activation(h)) if drop else self.activation(h)

        for ind in range(1, len(self.layers) - 1):
            h = self.layers[ind](h, batch_adjs[ind])
            h = self.batch_norms[ind](h) if self.use_batch_norms else h
            h = self.drop(self.activation(h)) if drop else self.activation(h)

        # The final adjacency rows follow the requested unlabeled target order.
        return self.proj_head(torch.sparse.mm(batch_adjs[-1], h))

    @torch.no_grad()
    def _get_biased_subgraphs(self, init_batch: np.ndarray,
                               batch_sizes: typing.List[int],
                               csr_adj_mat: sp.csr_matrix,
                               random_rng: bool = False,
                               contrastive_batch: np.ndarray = None) -> tuple:
        adj_out_gcn, alpha_out_gcn = self.get_subgraphs_concated_sampling(init_batch, batch_sizes, csr_adj_mat, random_rng=random_rng)
        if contrastive_batch is not None:
            adj_out_nce, alpha_out_nce = self.get_subgraphs_concated_sampling(contrastive_batch, batch_sizes, csr_adj_mat, random_rng=True)
            return (adj_out_gcn, alpha_out_gcn), (adj_out_nce, alpha_out_nce)

        return adj_out_gcn, alpha_out_gcn

    @torch.no_grad()
    def init_norm_cache(self, batch_size: int = 1024) -> None:
        N      = self.precompute.shape[0]
        layer0 = self.layers[0]
        print(f"[CACHE] Initialising norm cache for {N} nodes …")
        for start in range(0, N, batch_size):
            idx  = np.arange(start, min(start + batch_size, N))
            data = self.precompute[idx].to(self.device)
            h    = self.activation(layer0.precomputed_forward(data))
            self.norm_cache.update(idx, h)
        self.norm_cache.step()
        print("[CACHE] Done.")

