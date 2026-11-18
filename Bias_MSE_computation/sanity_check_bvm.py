"""Small dependency-light sanity check for the corrected BVM diagnostic.

This script does not require torch_geometric or OGB. It verifies that, for a
one-hidden-layer frozen model under i.i.d. sampling with replacement:

1. alpha=1 has near-zero Monte Carlo-corrected bias;
2. alpha* reduces variance relative to alpha=1;
3. the empirical MSE decomposition is numerically exact.
"""
import numpy as np
import scipy.sparse as sp
import torch

from model import BVSampler, HistoricalNormCache
from utility import normalize_adj, estimate_sampling_bias_variance_mse


def build_graph(num_nodes: int = 20) -> sp.csr_matrix:
    rows, cols = [], []
    for i in range(num_nodes):
        for j in {i, (i - 1) % num_nodes, (i + 1) % num_nodes,
                  (i + 3) % num_nodes, (i + 7) % num_nodes}:
            rows.extend([i, j])
            cols.extend([j, i])
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )
    adj.data[:] = 1.0
    return normalize_adj(adj)


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    num_nodes = 20
    adj = build_graph(num_nodes)
    x = torch.randn(num_nodes, 5)
    cache = HistoricalNormCache(num_nodes=num_nodes)
    model = BVSampler(
        input_dim=5,
        hidden_dims=[4],
        output_dim=3,
        csr_mat=adj,
        x=x,
        norm_cache=cache,
        samp_probs=np.ones(num_nodes),
        device=torch.device("cpu"),
        alpha_mode="optimal",
    )
    model.init_norm_cache()

    common = dict(
        model=model,
        csr_mat=adj,
        eval_nodes=np.arange(5),
        batch_sizes=[5, 3],
        num_mc_samples=2000,
        eval_batch_size=5,
        sampling_seed=7,
        replacement=True,
    )
    unbiased = estimate_sampling_bias_variance_mse(
        **common, alpha_mode="one"
    )
    optimal = estimate_sampling_bias_variance_mse(
        **common, alpha_mode="optimal"
    )

    print("alpha=1:", unbiased)
    print("alpha*= :", optimal)

    assert unbiased["normalized_bias2_mc_corrected"] < 0.01
    assert optimal["normalized_variance"] < unbiased["normalized_variance"]
    assert unbiased["decomposition_gap"] < 1e-6
    assert optimal["decomposition_gap"] < 1e-6
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
