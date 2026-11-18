#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import argparse
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import os
import csv
import torch.nn as nn
from model import BVSampler, InfoNCELoss, HistoricalNormCache
from utility import *

# --dataset Cora --norm_feat true --depths 1 2 3 4  --hidden_dim 16 --init_batch 256 --sample_size 100 --epochs 1000 --early_stop 20 --alpha_mode optimal --sampling_replacement false --compute_bvm true --bvm_compare_alpha true --bvm_replacement true --bvm_samples 100 --bvm_nodes 500  --bvm_output bvm_cora_paired_corrected.csv

def set_seeds(seed: int = 123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed=seed)


def model_train(
    model            : BVSampler,
    opt              : torch.optim.Optimizer,
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
    con_weight       : float,
) -> list:
    """
    One training step.

    Loss = L_CE(labeled_batch)  +  λ_c · L_InfoNCE(unlabeled_batch)

    The labeled and unlabeled portions of init_batch are always at known
    offsets (labeled first, unlabeled after), making it trivial to split
    the logits returned by forward() for the CE loss.

    InfoNCE compares the sampled aggregations of the same unlabeled targets
    from the mixed-batch view and a separately sampled view.
    """
    model.train()
    opt.zero_grad()

    if stoch:
        lab_bat, unlab_bat, init_bat = compose_mixed_batch(
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            batch_size=batch_list[0],
            labeled_ratio=labeled_ratio,
            rng=model.rng)
    else:
        init_bat  = None
        lab_bat   = None
        unlab_bat = np.array([], dtype=np.int64)

    # Determine whether InfoNCE is needed this step so we can request the
    # contrastive subgraph from forward() in the same sampling call.
    _need_contrastive = stoch and len(unlab_bat) > 1 and con_weight > 0.0
    out, contra_h, _, nce_subgraph = model(
        csr_mat           = csr_mat,
        drop              = True,
        stochastic        = stoch,
        batch_sizes       = batch_list,
        init_batch        = init_bat,
        contrastive_batch = unlab_bat if _need_contrastive else None)

    if stoch:
        # Labeled nodes occupy the first len(lab_bat) rows of `out`
        n_lab  = len(lab_bat)
        labels = y[lab_bat]
        l_ce   = loss_fn(out[:n_lab], labels)
    else:
        l_ce = loss_fn(out[training_mask], y[training_mask])

    total_loss = l_ce

    if _need_contrastive:
        z1 = model.proj_head(contra_h[n_lab:])
        adj_out_nce, alpha_out_nce = nce_subgraph
        z2       = model.forward_embed_presampled(adj_out_nce, alpha_out_nce, drop=True)
        l_con    = con_weight * model.infoNCELoss(z1, z2)
        total_loss = total_loss + l_con

    total_loss.backward()
    opt.step()
    losses.append(total_loss.item())
    return losses

cs = {"orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "red": "#D55E00", "pink": "#CC79A7", "black": "#000000"}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='BV-Sampler')
    parser.add_argument('--dataset', type=str, default="Cora", choices=["Cora", "PubMed", "CiteSeer", "Reddit", "ogbn-arxiv", "ogbn-products"],help='Dataset to use.')
    parser.add_argument('--norm_feat', type=str, default='false', choices=['true', 'false'],
                        help='Normalized features?')
    parser.add_argument('--batch_norm', type=str, default='false', choices=['true', 'false'],
                        help='Use batch normalization?')
    parser.add_argument('--report', type=int, default=1, help='How often to report accuracies; for bigger data'
                                                              'it may be better to take more GD steps first.')
    # METHOD + ARCHITECTURE
    parser.add_argument('--fast', type=str, default="true", choices=["true", "false"],
                        help='Use FastGCN or regular GCN.')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Dimension of the hidden layer.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--init_batch', type=int, default=256, help='Initial batch size.')
    parser.add_argument('--sample_size', type=int, default=400, help='Sample size size.')
    parser.add_argument('--scale_factor', type=float, default=1, help='For deeper networks, we need more samples.')
    parser.add_argument('--samp_dist', type=str, default='importance', choices=['importance', 'uniform'],
                        help='Which sampling distribution to use.')
    parser.add_argument('--alpha_mode', type=str, default='optimal', choices=['optimal', 'one'],
                        help='Alpha mode used during training.')
    parser.add_argument('--sampling_replacement', type=str, default='false', choices=['true', 'false'],
                        help='Use i.i.d. sampling with replacement during training.')
    # TRAINING
    parser.add_argument('--epochs', type=int, default=800, help='Total number of updates rounds.')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam learning rate.')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping term.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (l2 regularization).')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate.')
    # INFERENCE
    parser.add_argument('--samp_inference', type=str, default='false', choices=['true', 'false'],
                        help='Sample during inference phase for testing accuracy?')
    parser.add_argument('--num_samp_inference', type=int, default=1,
                        help='Number of times to sample during inference.')
    # LABEL-SCARCE SETTING
    parser.add_argument('--label_rate', type=float, default=1.0, help='Fraction of training labels to keep (1.0 = all).')
    parser.add_argument('--labeled_ratio', type=float, default=1.0, help='Target labeled fraction of each mini-batch (default: 1.0).')
    # CONTRASTIVE LEARNING
    parser.add_argument('--proj_dim', type=int, default=50, help='Projection head output dimension.')
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature τ. Smaller = harder negatives.')
    parser.add_argument('--con_weight', type=float, default=0.5, help='Loss weight λ_c for the unlabeled InfoNCE term. Set to 0 to disable contrastive loss.')
    # EXTRAS
    parser.add_argument('--use_cuda', type=str, default="true", choices=['true', 'false'],
                        help='Number of times to sample during inference.')
    parser.add_argument('--save_results', type=int, default=0, choices=[0, 1],
                        help='Save results or not (0 = do NOT save, 1 = save).')
    parser.add_argument('--seed', type=int, help='Run this training seed only. Defaults to 42, 43, and 44.')
    # DEPTH / BIAS-VARIANCE DIAGNOSTICS
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Numbers of hidden GCN layers to evaluate.')
    parser.add_argument('--compute_bvm', type=str, default='true', choices=['true', 'false'],
                        help='Estimate final-aggregation bias, variance, and MSE after training.')
    parser.add_argument('--bvm_samples', type=int, default=20,
                        help='Number of Monte Carlo sampled forward passes.')
    parser.add_argument('--bvm_nodes', type=int, default=2000,
                        help='Maximum number of fixed evaluation nodes used by the diagnostic.')
    parser.add_argument('--bvm_split', type=str, default='validation', choices=['validation', 'test'],
                        help='Node split used by the bias-variance diagnostic.')
    parser.add_argument('--bvm_seed', type=int, default=2026,
                        help='Seed for node selection and paired Monte Carlo sampling.')
    parser.add_argument('--bvm_replacement', type=str, default='true', choices=['true', 'false'],
                        help='Use i.i.d. sampling with replacement for the theory diagnostic.')
    parser.add_argument('--bvm_compare_alpha', type=str, default='true', choices=['true', 'false'],
                        help='Evaluate alpha=1 and alpha* on the same frozen model and paired draws.')
    parser.add_argument('--bvm_output', type=str, default='',
                        help='CSV output path; empty chooses a dataset-specific filename.')
    args = parser.parse_args()

    # Get the device
    user_device = torch.device("cuda:0") if torch.cuda.is_available() and args.use_cuda == 'true' else torch.device("cpu")
    args.fast = True if args.fast == 'true' else False
    args.batch_norm = True if args.batch_norm == 'true' else False
    args.samp_inference = True if args.samp_inference == 'true' else False
    args.norm_feat = True if args.norm_feat == 'true' else False
    args.compute_bvm = True if args.compute_bvm == 'true' else False
    args.sampling_replacement = args.sampling_replacement == 'true'
    args.bvm_replacement = args.bvm_replacement == 'true'
    args.bvm_compare_alpha = args.bvm_compare_alpha == 'true'
    args.early_stop = args.epochs + 1 if args.early_stop <= 0 else args.early_stop

    # Set the architecture

    score_list = []
    bvm_records = []
    total_epochs = args.epochs
    base_early_stop = args.early_stop
    base_hidden_dim = args.hidden_dim
    os.makedirs('tmp', exist_ok=True)

    # L is the number of hidden GCN layers. Keep the scalar hidden width fixed
    # instead of repeatedly wrapping args.hidden_dim into nested lists.
    for L in args.depths:
        hidden_dims = [base_hidden_dim] * L
        print('\n\n\nL=', L)
        args.num_layers = L
        args.epochs = total_epochs
        args.early_stop = base_early_stop
        if args.init_batch < 200:
            args.epochs = int(args.epochs * 256 // args.init_batch)
            args.early_stop = int(args.early_stop * 256 // args.init_batch)
            if args.epochs > 10000:
                args.epochs = 10000
        args.report = 50
        for rd_seed in ([42, 43, 44] if args.seed is None else [args.seed]):
            set_seeds(rd_seed)
            # Load the data - ToUndirected ensures that we can scan edge_list[0, :] to get all of the neighbors
            if args.dataset in ['ogbn-arxiv', 'ogbn-products']:
                dataset = PygNodePropPredDataset(name=args.dataset, root='../data', transform=T.ToSparseTensor())
                data = dataset[0]
                X = data.x.to(user_device)
                y = data.y.flatten().to(user_device)
                X = row_normalize(X) if args.norm_feat else X
                try:
                    adjmat = sp.load_npz(f"../data/{args.dataset}_adjmat.npz")
                except:
                    csr_edge_list = data.adj_t.to_symmetric().to_scipy().tocsr()
                    csr_edge_list += sp.identity(X.shape[0], format='csr')
                    # Normalize the adjacency matrix
                    adjmat = normalize_adj(csr_edge_list)
                    sp.save_npz(f"../data/{args.dataset}_adjmat.npz", adjmat)
                try:
                    fast_gcn_probs = np.load(f"../data/{args.dataset}_probs.npz")
                except:
                    fast_gcn_probs = np.asarray(adjmat.sum(1)).flatten()
                    np.save(f"../data/{args.dataset}_probs.npz", fast_gcn_probs)
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
                csr_edge_list = sp.csr_matrix((np.ones(data.edge_index.shape[1]),  # data
                                               (numpy_edges[0], numpy_edges[1])),  # (row, col)
                                              shape=(X.shape[0], X.shape[0]))  # size
                csr_edge_list += sp.identity(X.shape[0], format='csr')
                # Calculate adjacency matrix and probabilities
                adjmat = normalize_adj(csr_edge_list)
                fast_gcn_probs = np.asarray(adjmat.sum(1)).flatten()
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
                    np.savez(f"../data/{args.dataset}_{args.label_rate}_label.npz", train_test_data_split,
                             allow_pickle=True)
                training_indices = labeled_indices
                testing_indices = np.concatenate((testing_indices, unlabeled_indices), axis=0)
            else:
                # unlabeled_indices =torch.cat((torch.LongTensor(validation_indices), torch.LongTensor(testing_indices)))
                unlabeled_indices = np.setdiff1d(np.arange(len(y), dtype=np.int64), training_indices)
                labeled_indices = training_indices

            norm_cache = HistoricalNormCache(num_nodes=X.shape[0], init_norm=1.0, device=user_device)
            infonce_loss = InfoNCELoss(temperature=args.temperature).to(user_device)

            # Declare the model and optimizer
            model = BVSampler(input_dim=X.shape[1], hidden_dims=hidden_dims, output_dim=int(y.max().item()) + 1, dropout=args.drop,
                            csr_mat=adjmat, x=X, norm_cache=norm_cache,
                            samp_probs=np.ones((len(y),)) if args.samp_dist == 'uniform' else fast_gcn_probs,
                            device=user_device,
                            use_batch_norms=args.batch_norm,
                            dataset_name=args.dataset,
                            save_path='data', seed=rd_seed,
                            infoNCELoss=infonce_loss,
                            proj_dim=args.proj_dim,
                            alpha_mode=args.alpha_mode,
                            sampling_replacement=args.sampling_replacement,
                            )
            print(f"Your model:\n{model}")
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
            criteria = nn.NLLLoss(reduction='mean')
            batches = [args.init_batch] + [min(X.shape[0], int(args.sample_size * (1 if i == 0 else args.scale_factor))) for i in range(len(model.layers) - 1)]
            inference_batches = [args.init_batch] + [min(X.shape[0], int(args.sample_size * (1 if i == 0 else args.scale_factor))) for i in range(len(model.layers) - 1)]

            model.init_norm_cache(batch_size=1024)

            # Save meaningful results
            loss_hist = []
            val_hist = []
            test_acc = []

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
            best_iter = 0
            valacc_list = []
            patience = 0
            checkpoint_path = (
                f"tmp/{args.dataset}_L{L}_seed{rd_seed}_"
                f"trainalpha-{args.alpha_mode}_repl-{int(args.sampling_replacement)}.pt"
            )
            for i in range(1, args.epochs + 1):
                t0 = time.time()
                # loss_hist = model_train(model, optimizer, y, adjmat, training_indices, criteria, stochastic, loss_hist, batches)
                loss_hist = model_train(model, optimizer, y, adjmat, training_indices, criteria, stochastic, loss_hist,
                                        batches, labeled_indices, unlabeled_indices, args.labeled_ratio, args.con_weight)
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
                    # No sampling
                    else:
                        test_acc = test(model, X, y, adjmat, testing_indices, test_acc)
                        valacc_list = test(model, X, y, adjmat, validation_indices, valacc_list)
                    val_acc = valacc_list[-1]
                    # Check the validation performance ONLY if we are not performing sampled inference
                    max_acc = max(test_acc)
                    print(f"[{i:04d}] loss={loss_hist[-1]:.4f}  val_acc={val_acc:.2f}%  test_acc={test_acc[-1]:.2f}%  ")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_iter = i
                        patience = 0
                        torch.save(model.state_dict(), checkpoint_path)
                        print(f"Found best acc ({val_acc:.5f}) at iteration of {best_iter}. Saving best model...")
                    else:
                        patience += 1

                    if patience >= args.early_stop:
                        break

            if os.path.exists(checkpoint_path):
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=user_device)
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

            if args.compute_bvm:
                eval_pool = validation_indices if args.bvm_split == 'validation' else testing_indices
                eval_pool = np.asarray(eval_pool, dtype=np.int64)
                eval_rng = np.random.default_rng(args.bvm_seed)
                if len(eval_pool) > args.bvm_nodes:
                    eval_nodes = np.sort(
                        eval_rng.choice(eval_pool, size=args.bvm_nodes, replace=False)
                    )
                else:
                    eval_nodes = np.sort(eval_pool)

                # Compare alpha=1 and alpha* on exactly the same frozen weights.
                # estimate_sampling_bias_variance_mse resets the sampler to the
                # same seed for each mode, making the sampled neighborhoods paired.
                eval_alpha_modes = (
                    ["one", "optimal"] if args.bvm_compare_alpha
                    else [args.alpha_mode]
                )
                for eval_alpha_mode in eval_alpha_modes:
                    bvm = estimate_sampling_bias_variance_mse(
                        model=model,
                        csr_mat=adjmat,
                        eval_nodes=eval_nodes,
                        batch_sizes=inference_batches,
                        num_mc_samples=args.bvm_samples,
                        eval_batch_size=args.init_batch,
                        sampling_seed=args.bvm_seed,
                        alpha_mode=eval_alpha_mode,
                        replacement=args.bvm_replacement,
                    )
                    print_bias_variance_report(
                        bvm,
                        prefix=f"[BVM L={L} seed={rd_seed}]",
                    )

                    record = {
                        "dataset": args.dataset,
                        "depth": L,
                        "seed": rd_seed,
                        "split": args.bvm_split,
                        "train_alpha_mode": args.alpha_mode,
                        "train_sampling_replacement": args.sampling_replacement,
                        "eval_alpha_mode": eval_alpha_mode,
                        "test_micro_f1": final_test_acc[-1],
                        **bvm,
                    }
                    bvm_records.append(record)



    if args.compute_bvm and bvm_records:
        output_path = args.bvm_output or (
            f"bvm_{args.dataset}_train-{args.alpha_mode}_paired.csv"
        )
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bvm_records[0].keys()))
            writer.writeheader()
            writer.writerows(bvm_records)
        print(f"[BVM] Saved {len(bvm_records)} depth/seed records to {output_path}")
