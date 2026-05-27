#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import argparse
import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import os
import torch.nn as nn
from model import BVSampler, InfoNCELoss, HistoricalNormCache
from utility import *

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

    The InfoNCE loss is computed by calling forward_embed() *twice* on the
    unlabeled nodes, producing two distinct neighbourhood views from
    independent sampler draws — no separate augmentation needed.
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
        drop              = False,
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
        z1 = model.proj_head(contra_h)
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
    # TRAINING
    parser.add_argument('--epochs', type=int, default=200, help='Total number of updates rounds.')
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
    parser.add_argument('--labeled_ratio', type=float, default=1.0, help='p: fraction of each mini-batch drawn from labeled nodes (default 0.80 → 80%% labeled, 20%% unlabeled).')
    # CONTRASTIVE LEARNING
    parser.add_argument('--proj_dim', type=int, default=50, help='Projection head output dimension.')
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature τ. Smaller = harder negatives.')
    parser.add_argument('--con_weight', type=float, default=0.5, help='Loss weight λ_c for the unlabeled InfoNCE term. Set to 0 to disable contrastive loss.')
    # EXTRAS
    parser.add_argument('--use_cuda', type=str, default="true", choices=['true', 'false'],
                        help='Number of times to sample during inference.')
    parser.add_argument('--save_results', type=int, default=0, choices=[0, 1],
                        help='Save results or not (0 = do NOT save, 1 = save).')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    # Get the device
    user_device = torch.device("cuda:0") if torch.cuda.is_available() and args.use_cuda == 'true' else torch.device("cpu")
    args.fast = True if args.fast == 'true' else False
    args.batch_norm = True if args.batch_norm == 'true' else False
    args.samp_inference = True if args.samp_inference == 'true' else False
    args.norm_feat = True if args.norm_feat == 'true' else False
    args.early_stop = args.epochs + 1 if args.early_stop <= 0 else args.early_stop

    # Set the architecture
    args.hidden_dim = [args.hidden_dim] * args.num_layers
    score_list = []
    # for rd_seed in [42, 43, 44, 45, 46]:
    for rd_seed in [43]:
        set_seeds(rd_seed)
        # Load the data - ToUndirected ensures that we can scan edge_list[0, :] to get all of the neighbors
        if args.dataset in ['ogbn-arxiv', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=args.dataset, root='./data', transform=T.ToSparseTensor())
            data = dataset[0]
            X = data.x.to(user_device)
            y = data.y.flatten().to(user_device)
            X = row_normalize(X) if args.norm_feat else X
            try:
                adjmat = sp.load_npz(f"./data/{args.dataset}_adjmat.npz")
            except:
                csr_edge_list = data.adj_t.to_symmetric().to_scipy().tocsr()
                csr_edge_list += sp.identity(X.shape[0], format='csr')
                # Normalize the adjacency matrix
                adjmat = normalize_adj(csr_edge_list)
                sp.save_npz(f"./data/{args.dataset}_adjmat.npz", adjmat)
            try:
                fast_gcn_probs = np.load(f"./data/{args.dataset}_probs.npz")
            except:
                fast_gcn_probs = np.asarray(adjmat.sum(1)).flatten()
                np.save(f"./data/{args.dataset}_probs.npz", fast_gcn_probs)
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
            dataset = Reddit(root='./data', transform=T.ToSparseTensor(remove_edge_index=False)) \
                if args.dataset == 'Reddit' \
                else Planetoid(root='./data', name=args.dataset, transform=T.ToSparseTensor(remove_edge_index=False))
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
            if os.path.exists(f"./data/{args.dataset}_{args.label_rate}_label.npz"):
                with np.load(f"./data/{args.dataset}_{args.label_rate}_label.npz", allow_pickle=True) as loaded:
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
                np.savez(f"./data/{args.dataset}_{args.label_rate}_label.npz", train_test_data_split,
                         allow_pickle=True)
            training_indices = labeled_indices
            testing_indices = np.concatenate((testing_indices, unlabeled_indices), axis=0)

        norm_cache = HistoricalNormCache(num_nodes=X.shape[0], init_norm=1.0, device=user_device)
        infonce_loss = InfoNCELoss(temperature=args.temperature).to(user_device)

        # Declare the model and optimizer
        model = BVSampler(input_dim=X.shape[1], hidden_dims=args.hidden_dim, output_dim=max(y).item() + 1, dropout=args.drop,
                        csr_mat=adjmat, x=X, norm_cache=norm_cache,
                        samp_probs=np.ones((len(y),)) if args.samp_dist == 'uniform' else fast_gcn_probs,
                        device=user_device,
                        use_batch_norms=args.batch_norm,
                        dataset_name=args.dataset,
                        save_path='data', seed=args.seed,
                        infoNCELoss = infonce_loss,
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
        valacc_list = []
        patience = 0
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
                    torch.save(model.state_dict(), "tmp/fastgcn_redditModel.pt")
                    print(f"Found best acc ({val_acc:.5f}) at iteration of {best_iter}. Saving best model...")
                else:
                    patience += 1

                if patience >= args.early_stop:
                    break

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
