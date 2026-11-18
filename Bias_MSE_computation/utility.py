import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score as F1


def test(model, x, y, csr_mat, mask, accuracy):
    model.eval()
    pred, _ = model.predict(x, csr_mat)
    accuracy.append(F1(y[mask].cpu().numpy(), pred[mask].cpu().numpy(),
                       average='micro') * 100.0)
    return accuracy


def sample_test(model, x, y, csr_mat, batch_list, num_samp_inf, mask, accuracy):
    model.eval()
    total_out, total_y = np.array([]), np.array([])
    for i in range(0, len(mask), batch_list[0]):
        m = mask[i:i + batch_list[0]]
        out, _, ib = model.sample_predict(csr_mat, m, batch_list, num_samp_inf)
        total_out = np.concatenate((total_out, out.cpu().numpy())).flatten()
        total_y   = np.concatenate((total_y, y[ib].cpu().numpy())).flatten()
    accuracy.append(F1(total_y, total_out, average='micro') * 100.0)
    return accuracy

def _csr_to_torch(adj) -> torch.Tensor:
    if sp.isspmatrix_coo(adj) is False:
        adj = sp.coo_matrix(adj)
    return torch.sparse_coo_tensor(np.array([adj.row, adj.col]), adj.data.astype(np.float32), adj.shape, dtype=torch.float32)


def normalize_adj(adj) -> sp.csr_matrix:
    adj = sp.coo_matrix(adj)
    d   = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d[np.isinf(d)] = 0.
    D   = sp.diags(d)
    return adj.dot(D).transpose().dot(D).tocsr()


def row_normalize(mat: torch.Tensor) -> torch.Tensor:
    r = 1. / torch.norm(mat, dim=1, p=2)
    r[r == torch.inf] = 0.
    return mat * r.unsqueeze(1)


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

    n_lab = max(1, int(round(batch_size * labeled_ratio)))
    n_unlab = min(batch_size - n_lab, len(unlabeled_indices))
    n_lab = min(n_lab, len(labeled_indices))
    lab_bat   = rng.choice(labeled_indices, size=n_lab, replace=False)
    unlab_bat = rng.choice(unlabeled_indices, size=n_unlab, replace=False)

    init_bat  = np.concatenate([lab_bat, unlab_bat])

    return lab_bat, unlab_bat, init_bat


@torch.no_grad()
def estimate_sampling_bias_variance_mse(
    model,
    csr_mat: sp.csr_matrix,
    eval_nodes,
    batch_sizes,
    num_mc_samples: int = 20,
    eval_batch_size: int = None,
    sampling_seed: int = 2026,
    alpha_mode: str = None,
    replacement: bool = True,
    eps: float = 1e-12,
) -> dict:
    """Estimate squared bias, variance, and MSE of sampled aggregation.

    The exact reference is the full-graph final-layer aggregation ``A H`` for the
    same fixed nodes. The raw squared-bias estimate uses the Monte Carlo sample
    mean and therefore contains a finite-R contribution of approximately
    ``Var/R``. ``bias2_mc_corrected`` subtracts an unbiased estimate of this
    contribution, while the raw quantities retain the exact empirical identity
    ``mse = bias2 + variance``.

    ``alpha_mode`` is an evaluation-time override, so ``one`` and ``optimal`` can
    be evaluated on the same frozen model. Resetting the same sampling seed in
    separate calls yields paired sampled neighborhoods.
    """
    if num_mc_samples < 2:
        raise ValueError("num_mc_samples must be at least 2 to estimate variance.")
    if alpha_mode is not None and alpha_mode not in {"optimal", "one"}:
        raise ValueError("alpha_mode must be either 'optimal', 'one', or None.")

    model.eval()
    nodes = np.asarray(eval_nodes, dtype=np.int64)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("eval_nodes must be a non-empty one-dimensional array.")

    eval_batch_size = int(eval_batch_size or batch_sizes[0])
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")

    exact_all = model.full_final_aggregation(csr_mat, nodes).detach()
    model.reset_sampling_seed(sampling_seed)

    totals = {
        "bias2": 0.0,
        "bias2_mc_corrected": 0.0,
        "variance": 0.0,
        "mse": 0.0,
        "energy": 0.0,
        "norm_ratio": 0.0,
        "projected_bias": 0.0,
        "num_nodes": 0,
    }
    all_alpha_values = []

    for start in range(0, nodes.size, eval_batch_size):
        stop = min(start + eval_batch_size, nodes.size)
        batch_nodes = nodes[start:stop]
        exact = exact_all[start:stop]

        draws = []
        for _ in range(num_mc_samples):
            sampled, sampling_stats = model.sampled_final_aggregation(
                csr_mat=csr_mat,
                init_batch=batch_nodes,
                batch_sizes=batch_sizes,
                alpha_mode=alpha_mode,
                replacement=replacement,
                return_sampling_stats=True,
            )
            draws.append(sampled.detach())
            for hop_stats in sampling_stats:
                values = hop_stats["alpha_values"]
                if values.numel() > 0:
                    all_alpha_values.append(values.to(torch.float64))

        samples = torch.stack(draws, dim=0)  # [R, B, D]
        mean_sample = samples.mean(dim=0)

        bias_vec = mean_sample - exact
        bias2_node = bias_vec.pow(2).sum(dim=1)
        variance_node = (samples - mean_sample.unsqueeze(0)).pow(2).sum(dim=2).mean(dim=0)
        mse_node = (samples - exact.unsqueeze(0)).pow(2).sum(dim=2).mean(dim=0)

        # Because variance_node uses denominator R, variance_node/(R-1) is an
        # unbiased estimate of Var(mean_sample)=Var(single draw)/R.
        mc_mean_noise_node = variance_node / float(num_mc_samples - 1)
        corrected_bias2_node = torch.clamp(bias2_node - mc_mean_noise_node, min=0.0)

        energy_node = exact.pow(2).sum(dim=1)
        norm_ratio_node = mean_sample.norm(dim=1) / (exact.norm(dim=1) + eps)
        projected_bias_node = (bias_vec * exact).sum(dim=1) / (energy_node + eps)

        totals["bias2"] += bias2_node.double().sum().item()
        totals["bias2_mc_corrected"] += corrected_bias2_node.double().sum().item()
        totals["variance"] += variance_node.double().sum().item()
        totals["mse"] += mse_node.double().sum().item()
        totals["energy"] += energy_node.double().sum().item()
        totals["norm_ratio"] += norm_ratio_node.double().sum().item()
        totals["projected_bias"] += projected_bias_node.double().sum().item()
        totals["num_nodes"] += len(batch_nodes)

        del samples, draws, sampled, mean_sample

    n = totals["num_nodes"]
    mean_energy = totals["energy"] / n
    bias2 = totals["bias2"] / n
    bias2_mc_corrected = totals["bias2_mc_corrected"] / n
    variance = totals["variance"] / n
    mse = totals["mse"] / n
    expected_mc_bias_noise = variance / float(num_mc_samples - 1)

    if all_alpha_values:
        alpha_values = torch.cat(all_alpha_values)
        alpha_mean = alpha_values.mean().item()
        alpha_median = alpha_values.median().item()
        alpha_p10 = torch.quantile(alpha_values, 0.10).item()
        alpha_p90 = torch.quantile(alpha_values, 0.90).item()
        num_alpha_values = int(alpha_values.numel())
    else:
        alpha_mean = alpha_median = alpha_p10 = alpha_p90 = 1.0
        num_alpha_values = 0

    result = {
        "num_eval_nodes": n,
        "num_mc_samples": int(num_mc_samples),
        "eval_alpha_mode": alpha_mode or model.alpha_mode,
        "sampling_replacement": bool(replacement),
        "bias2": bias2,
        "bias2_mc_corrected": bias2_mc_corrected,
        "variance": variance,
        "mse": mse,
        "mean_exact_energy": mean_energy,
        "normalized_bias2": bias2 / max(mean_energy, eps),
        "normalized_bias2_mc_corrected": bias2_mc_corrected / max(mean_energy, eps),
        "normalized_variance": variance / max(mean_energy, eps),
        "normalized_mse": mse / max(mean_energy, eps),
        "expected_mc_bias_noise": expected_mc_bias_noise,
        "normalized_expected_mc_bias_noise": expected_mc_bias_noise / max(mean_energy, eps),
        "bias2_to_mc_noise_ratio": bias2 / max(expected_mc_bias_noise, eps),
        "mean_norm_ratio": totals["norm_ratio"] / n,
        "mean_projected_bias": totals["projected_bias"] / n,
        "mean_alpha": alpha_mean,
        "median_alpha": alpha_median,
        "alpha_p10": alpha_p10,
        "alpha_p90": alpha_p90,
        "num_alpha_values": num_alpha_values,
        "decomposition_gap": abs(mse - (bias2 + variance)),
    }
    return result


def print_bias_variance_report(metrics: dict, prefix: str = "[BVM]") -> None:
    """Print a compact, reproducible bias-variance report."""
    print(
        f"{prefix} alpha={metrics['eval_alpha_mode']}  "
        f"replacement={metrics['sampling_replacement']}  "
        f"nodes={metrics['num_eval_nodes']}  MC={metrics['num_mc_samples']}\n"
        f"{prefix} raw bias^2={metrics['bias2']:.6e}  "
        f"MC-corrected bias^2={metrics['bias2_mc_corrected']:.6e}  "
        f"variance={metrics['variance']:.6e}  mse={metrics['mse']:.6e}\n"
        f"{prefix} normalized corrected bias^2="
        f"{metrics['normalized_bias2_mc_corrected']:.6e}  "
        f"normalized variance={metrics['normalized_variance']:.6e}  "
        f"normalized mse={metrics['normalized_mse']:.6e}\n"
        f"{prefix} alpha mean/median/p10/p90="
        f"{metrics['mean_alpha']:.6f}/{metrics['median_alpha']:.6f}/"
        f"{metrics['alpha_p10']:.6f}/{metrics['alpha_p90']:.6f}\n"
        f"{prefix} norm ratio={metrics['mean_norm_ratio']:.6f}  "
        f"projected bias={metrics['mean_projected_bias']:.6f}  "
        f"bias/noise ratio={metrics['bias2_to_mc_noise_ratio']:.3f}  "
        f"decomposition gap={metrics['decomposition_gap']:.3e}"
    )

