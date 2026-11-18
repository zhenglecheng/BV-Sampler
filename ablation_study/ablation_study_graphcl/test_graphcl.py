"""Small-graph checks; no dataset downloads or optional PyG extensions."""
import ast
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from model import OptISGCN, GraphCLLoss, HistoricalNormCache
from utility import compose_mixed_batch, sample_test


class GraphCLChecks(unittest.TestCase):
    def make_model(self, hidden=(8, 8), batch_norm=False):
        torch.manual_seed(42)
        self.adj = sp.csr_matrix(np.ones((24, 24), dtype=np.float32) / 24)
        self.x = torch.rand(24, 6)
        return OptISGCN(6, list(hidden), 3, self.adj, self.x,
                        samp_probs=np.ones(24), norm_cache=HistoricalNormCache(24),
                        proj_dim=5, infoNCELoss=GraphCLLoss(),
                        use_batch_norms=batch_norm)

    def test_loss_reference_and_gradients(self):
        a = torch.randn(3, 5, requires_grad=True)
        b = torch.randn(3, 5, requires_grad=True)
        z = torch.nn.functional.normalize(torch.cat([a, b]), dim=1)
        terms = []
        for i in range(6):
            scores = torch.stack([z[i].dot(z[j]) / .3 for j in range(6) if j != i])
            terms.append(torch.logsumexp(scores, 0) - z[i].dot(z[(i+3) % 6]) / .3)
        loss = GraphCLLoss(.3)(a, b)
        torch.testing.assert_close(loss, torch.stack(terms).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(ValueError):
            GraphCLLoss()(a[:1], b[:1])

    def test_mask_equivalence_and_target_alignment(self):
        for hidden in [(8,), (8, 8), (8, 8, 8)]:
            model = self.make_model(hidden, batch_norm=True)
            self.assertFalse(hasattr(model, 'precompute'))
            model.init_norm_cache(batch_size=5)
            rows = np.array([2, 7, 11])
            mask = torch.tensor([1., 0, 1, 0, 1, 1])
            torch.testing.assert_close(model.first_aggregate(rows) * mask,
                                       torch.tensor(self.adj[rows].toarray()) @ (self.x * mask))
            batches = [4] + [5] * len(hidden)
            saved = batches.copy()
            _, _, _, sampled = model(self.adj, stochastic=True, batch_sizes=batches,
                                      init_batch=np.arange(4))
            self.assertEqual(batches, saved)
            model.feature_mask_rate = 0
            z1, z2 = model.graphcl_views(*sampled, np.array([2, 3]))
            self.assertEqual(tuple(z1.shape), (2, 5))
            torch.testing.assert_close(z1, z2)
            model.infoNCELoss(z1, z2).backward()
            self.assertIsNotNone(model.proj_head.net[0].weight.grad)

    def test_one_sample_per_optimizer_step_and_batch_evaluation(self):
        model = self.make_model()
        calls = []
        original = model.get_subgraphs_concated_sampling
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        model.get_subgraphs_concated_sampling = counted
        ns = dict(np=np, torch=torch, sp=sp, nn=torch.nn,
                  OptISGCN=OptISGCN, compose_mixed_batch=compose_mixed_batch)
        tree = ast.parse(Path(__file__).with_name('BVSampler_main.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'model_train')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'train', 'exec'), ns)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        y = torch.arange(24) % 3
        for weight in [.5, 0.]:
            before = len(calls)
            losses = ns['model_train'](model, optimizer, y, self.adj, np.arange(4),
                torch.nn.NLLLoss(), True, [], [4, 5, 6], np.arange(4), np.arange(4, 24), 1., weight)
            self.assertEqual(len(calls), before + 1)
            self.assertTrue(np.isfinite(losses[-1]))
        before = len(calls)
        score = sample_test(model, self.x, y, self.adj, [4, 5, 6], 1, np.arange(10), [])[-1]
        self.assertEqual(len(calls), before + 3)
        self.assertTrue(0 <= score <= 100)

    def test_scarce_pools_and_isolated_nodes(self):
        lab, unlab, batch = compose_mixed_batch(np.array([0]), np.array([1, 2]), 256, 1., np.random.default_rng(42))
        self.assertEqual((len(lab), len(unlab), len(batch)), (1, 2, 3))
        model = self.make_model()
        model.csr_mat = sp.eye(24, format='csr', dtype=np.float32)
        out, _, _, _ = model(model.csr_mat, stochastic=True, batch_sizes=[4, 5, 6], init_batch=np.arange(4))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
