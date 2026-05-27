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

    n_unlab   = max(1, int(round(batch_size * (1 - labeled_ratio))))
    lab_bat   = rng.choice(labeled_indices, size=batch_size, replace=False)
    unlab_bat = rng.choice(unlabeled_indices, size=n_unlab, replace=False)

    init_bat  = np.concatenate([lab_bat, unlab_bat])

    return lab_bat, unlab_bat, init_bat
