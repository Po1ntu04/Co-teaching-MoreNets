# -*- coding:utf-8 -*-
import argparse
import datetime
import json
import math
import os
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Subset
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
from utils.bmm import BetaMixture1D, loss_to_score
from utils.replay import PurifiedReplayBuffer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--result_dir", type=str, default="results/", help="dir to save result txt files")
    parser.add_argument("--noise_rate", type=float, default=0.2, help="corruption rate, should be less than 1")
    parser.add_argument("--forget_rate", type=float, default=None, help="forget rate")
    parser.add_argument("--noise_type", type=str, default="pairflip", help="[pairflip, symmetric]")
    parser.add_argument("--num_gradual", type=int, default=10, help="epochs for linear drop rate (Tk)")
    parser.add_argument("--exponent", type=float, default=1, help="exponent of the forget rate schedule (c)")
    parser.add_argument("--top_bn", action="store_true")
    parser.add_argument("--dataset", type=str, default="mnist", help="mnist, cifar10, or cifar100")
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4, help="how many subprocesses to use for data loading")
    parser.add_argument("--num_iter_per_epoch", type=int, default=400)
    parser.add_argument("--epoch_decay_start", type=int, default=80)
    #------------------------------------------------------------------------#
    parser.add_argument("--num_models", type=int, default=3, help="M: number of peer models (>=2)")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="SAM perturbation coefficient (rho)")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="aggregation mode for peer losses",
    )
    parser.add_argument(
        "--reliability_decay",
        type=float,
        default=0.6,
        help="decay factor applied when a model underperforms",
    )
    parser.add_argument(
        "--reliability_gap",
        type=float,
        default=2.0,
        help="accuracy gap (percentage points) to trigger reliability decay",
    )
    parser.add_argument("--reliability_min", type=float, default=0.1, help="minimum reliability lambda")
    parser.add_argument(
        "--drop_last",
        action="store_true",
        help="drop last incomplete batch (useful if batch size mismatch causes selection errors)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    # ------------------------------------------------------------------------
    # EM-style q configuration (soft responsibilities)
    parser.add_argument(
        "--q_mode",
        type=str,
        default="posterior",
        choices=["posterior", "loss", "bmm"],
        help="q computation: posterior (mixture), loss (sigmoid), or bmm (Beta Mixture Model)",
    )
    parser.add_argument("--bmm_max_iters", type=int, default=10, help="max EM iterations for BMM fitting")
    parser.add_argument("--bmm_warmup", type=int, default=5, help="epochs before enabling BMM (use loss-q during warmup)")
    parser.add_argument("--q_gamma", type=float, default=0.5, help="soft/hard mixing weight gamma")
    parser.add_argument("--q_ema", type=float, default=0.9, help="EMA for per-sample Q smoothing")
    parser.add_argument("--q_temp_max", type=float, default=2.0, help="max temperature for q")
    parser.add_argument("--q_temp_min", type=float, default=0.5, help="min temperature for q")
    parser.add_argument("--q_temp_warmup", type=int, default=10, help="warmup steps for q temperature")
    parser.add_argument("--q_overlap_threshold", type=float, default=0.9, help="overlap trigger threshold")
    parser.add_argument("--q_overlap_boost", type=float, default=0.2, help="temperature boost when overlap high")
    parser.add_argument("--q_loss_tau", type=str, default="median", help="loss-q pivot: median or mean")
    # ------------------------------------------------------------------------
    # Prior / pi_t update (slow variable for streaming)
    parser.add_argument("--pi_init", type=float, default=0.8, help="initial clean prior pi")
    parser.add_argument("--pi_ema", type=float, default=0.99, help="EMA for pi_t")
    parser.add_argument("--pi_beta_a", type=float, default=2.0, help="Beta prior a for pi")
    parser.add_argument("--pi_beta_b", type=float, default=2.0, help="Beta prior b for pi")
    # ------------------------------------------------------------------------
    # Replay buffer (stream-like stability)
    parser.add_argument("--replay_size", type=int, default=2000, help="max replay buffer size")
    parser.add_argument("--replay_ratio", type=float, default=0.25, help="replay sample ratio per batch")
    parser.add_argument("--replay_tau", type=float, default=0.8, help="Q threshold to push into replay (for legacy mode)")
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="legacy",
        choices=["legacy", "purified"],
        help="replay buffer mode: legacy (threshold-based) or purified (BMM-based)",
    )
    parser.add_argument("--replay_admission", type=float, default=0.7, help="BMM posterior threshold for admission (purified mode)")
    parser.add_argument("--replay_stability", type=int, default=3, help="required consecutive high-posterior epochs (purified mode)")
    parser.add_argument("--replay_evict", type=float, default=0.5, help="BMM posterior threshold for eviction (purified mode)")
    parser.add_argument("--replay_ema", type=float, default=0.3, help="EMA alpha for clean_p updates (purified mode)")
    parser.add_argument("--replay_sample_strategy", type=str, default="uniform", choices=["uniform", "weighted", "quality"], help="sampling strategy for purified replay")
    # ------------------------------------------------------------------------
    # Reliability / active set (soft absorb, optional prune)
    parser.add_argument("--lambda_active", type=float, default=0.2, help="lambda threshold for active models")
    parser.add_argument("--lambda_patience", type=int, default=5, help="patience before deactivating model")
    parser.add_argument("--min_active", type=int, default=2, help="minimum active models to keep")
    parser.add_argument("--val_split", type=float, default=0.1, help="validation split ratio from train set")
    # ------------------------------------------------------------------------
    # Explore sampling (diversity preservation when overlap is high)
    parser.add_argument("--explore_delta", type=float, default=0.0, help="fraction of batch for explore sampling (0=disabled)")
    parser.add_argument("--explore_trigger", type=float, default=0.85, help="overlap threshold to trigger explore sampling")
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_rate_schedule(
    forget_rate: float, num_gradual: int, exponent: float, n_epoch: int
) -> np.ndarray:
    schedule = np.ones(n_epoch) * forget_rate
    schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
    return schedule


def adjust_learning_rate(optimizer: torch.optim.Optimizer, alpha_plan: Sequence[float], beta1_plan: Sequence[float], epoch: int) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha_plan[epoch]
        param_group["betas"] = (beta1_plan[epoch], 0.999)  # Only change beta1


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100.0


def linear_anneal(start: float, end: float, step: int, warmup: int) -> float:
    if warmup <= 0:
        return end
    ratio = min(float(step) / float(warmup), 1.0)
    return start + (end - start) * ratio


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    weights = weights.detach()
    return (per_sample * weights).sum() / (weights.sum() + 1e-12)


def split_train_val(dataset, val_split: float, seed: int):
    if val_split <= 0:
        return dataset, None
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    indices = np.arange(n_total)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    return train_subset, val_subset


def aggregate_losses(
    loss_stack: torch.Tensor, model_idx: int, active_mask: torch.Tensor, mode: str = "mean"
) -> torch.Tensor:
    """
    Compute aggregated loss for clean selection using peers only (exclude model_idx).
    loss_stack: shape (M, B), reliability: shape (M,)
    """
    mask = active_mask.clone()
    if mask.numel() == 0:
        return loss_stack[model_idx]
    mask[model_idx] = False
    peer_losses = loss_stack[mask]  # shape (M-1, B)
    if peer_losses.numel() == 0:
        return loss_stack[model_idx]
    if mode == "median":
        return peer_losses.median(dim=0).values
    return peer_losses.mean(dim=0)


def sam_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    rho: float,
) -> Tuple[float, float]:
    """
    Two-step SAM update on the provided images/labels subset.
    Returns: (clean_loss, perturbed_loss)
    """
    optimizer.zero_grad()
    clean_loss = weighted_ce(model(images), labels, weights)
    clean_loss.backward()
    grad_norm = torch.norm(
        torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]),
        p=2,
    )
    scale = rho / (grad_norm + 1e-12)
    e_ws: List[torch.Tensor] = []
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                e_ws.append(None)
                continue
            e_w = p.grad * scale
            p.add_(e_w)
            e_ws.append(e_w)
    optimizer.zero_grad()
    perturbed_loss = weighted_ce(model(images), labels, weights)
    perturbed_loss.backward()
    with torch.no_grad():
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is None:
                continue
            p.sub_(e_w)
    optimizer.step()
    return clean_loss.item(), perturbed_loss.item()


def evaluate_models(models: List[torch.nn.Module], loader) -> Tuple[List[float], float]:
    for m in models:
        m.eval()
    total = 0
    correct_counts = [0 for _ in models]
    ensemble_correct = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True).long()
            logits_list = [m(images) for m in models]
            total += labels.size(0)
            # individual
            for idx, logits in enumerate(logits_list):
                preds = logits.argmax(dim=1)
                correct_counts[idx] += (preds == labels).sum().item()
            # ensemble majority vote
            stacked = torch.stack([l.softmax(dim=1) for l in logits_list], dim=0)
            ensemble_preds = stacked.mean(dim=0).argmax(dim=1)
            ensemble_correct += (ensemble_preds == labels).sum().item()
    accs = [100.0 * c / total for c in correct_counts]
    ensemble_acc = 100.0 * ensemble_correct / total
    return accs, ensemble_acc


def update_reliabilities(
    lambdas: List[float], accuracies: List[float], decay: float, gap: float, min_lambda: float
) -> List[float]:
    best = max(accuracies)
    updated = []
    for lam, acc in zip(lambdas, accuracies):
        if best - acc > gap:
            lam = max(min_lambda, lam * decay)
        updated.append(min(1.0, lam))
    return updated


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_datasets(args) -> Tuple:
    if args.dataset == "mnist":
        input_channel = 1
        num_classes = 10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = max(args.n_epoch, 200)
        train_dataset = MNIST(
            root="./data/",
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
        test_dataset = MNIST(
            root="./data/",
            download=True,
            train=False,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
    elif args.dataset == "cifar10":
        input_channel = 3
        num_classes = 10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = max(args.n_epoch, 200)
        train_dataset = CIFAR10(
            root="./data/",
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
        test_dataset = CIFAR10(
            root="./data/",
            download=True,
            train=False,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
    elif args.dataset == "cifar100":
        input_channel = 3
        num_classes = 100
        args.top_bn = False
        args.epoch_decay_start = 100
        args.n_epoch = max(args.n_epoch, 200)
        train_dataset = CIFAR100(
            root="./data/",
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
        test_dataset = CIFAR100(
            root="./data/",
            download=True,
            train=False,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
    else:
        raise ValueError("Unsupported dataset")
    return input_channel, num_classes, train_dataset, test_dataset


def train_epoch(
    epoch: int,
    args,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    loader,
    train_dataset,
    num_classes: int,
    remember_rate: float,
    reliability: List[float],
    active_mask: List[bool],
    q_global: np.ndarray,
    pi_t: float,
    replay_buffer: List[int],
    replay_set: set,
    bmm: BetaMixture1D = None,
    purified_replay: PurifiedReplayBuffer = None,
) -> Tuple[Dict[str, float], np.ndarray, float, List[int], set, BetaMixture1D, PurifiedReplayBuffer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in models:
        m.train()
    active_mask_tensor = torch.tensor(active_mask, device=device, dtype=torch.bool)
    
    # Initialize BMM if using bmm mode
    if bmm is None and args.q_mode == "bmm":
        bmm = BetaMixture1D(max_iters=args.bmm_max_iters)
    
    # Initialize purified replay if using purified mode
    if purified_replay is None and args.replay_mode == "purified" and args.replay_size > 0:
        purified_replay = PurifiedReplayBuffer(
            max_size=args.replay_size,
            admission_threshold=args.replay_admission,
            stability_threshold=args.replay_stability,
            evict_threshold=args.replay_evict,
            ema_alpha=args.replay_ema,
        )
    
    # Accumulate losses for epoch-level BMM fitting
    epoch_losses: List[np.ndarray] = []
    epoch_indices: List[np.ndarray] = []
    
    batch_accumulator: Dict[str, List[float]] = {
        "train_acc": [0.0 for _ in models],
        "clean_loss": [0.0 for _ in models],
        "sharp_loss": [0.0 for _ in models],
        "q_mean": [],
        "q_std": [],
        "overlap": [],
    }
    num_batches = 0

    for batch_idx, (images, labels, indices) in enumerate(loader):
        if batch_idx >= args.num_iter_per_epoch:
            break
        # --- replay sampling (supports both legacy and purified modes) ---
        if args.replay_mode == "purified" and purified_replay is not None:
            # Purified mode: sample from PurifiedReplayBuffer
            replay_count = max(0, int(len(labels) * args.replay_ratio))
            replay_count = min(replay_count, len(purified_replay))
            if replay_count > 0:
                replay_idx = purified_replay.sample(replay_count, strategy=args.replay_sample_strategy)
                base_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
                replay_samples = [base_dataset[i] for i in replay_idx]
                replay_imgs, replay_lbls, replay_ids = zip(*replay_samples)
                replay_imgs = torch.stack(list(replay_imgs), dim=0)
                replay_lbls = torch.tensor(replay_lbls, dtype=labels.dtype)
                replay_ids = torch.tensor(replay_ids, dtype=indices.dtype)
                images = torch.cat([images, replay_imgs], dim=0)
                labels = torch.cat([labels, replay_lbls], dim=0)
                indices = torch.cat([indices, replay_ids], dim=0)
        else:
            # Legacy mode: simple threshold-based replay
            if args.replay_ratio > 0 and replay_buffer:
                replay_count = max(0, int(len(labels) * args.replay_ratio))
                replay_count = min(replay_count, len(replay_buffer))
            else:
                replay_count = 0
            if replay_count > 0:
                replay_idx = np.random.choice(replay_buffer, size=replay_count, replace=False)
                base_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
                replay_samples = [base_dataset[i] for i in replay_idx]
                replay_imgs, replay_lbls, replay_ids = zip(*replay_samples)
                replay_imgs = torch.stack(list(replay_imgs), dim=0)
                replay_lbls = torch.tensor(replay_lbls, dtype=labels.dtype)
                replay_ids = torch.tensor(replay_ids, dtype=indices.dtype)
                images = torch.cat([images, replay_imgs], dim=0)
                labels = torch.cat([labels, replay_lbls], dim=0)
                indices = torch.cat([indices, replay_ids], dim=0)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True).long()
        indices = indices.cuda(non_blocking=True)
        batch_size = labels.size(0)

        logits_list = [m(images) for m in models]
        loss_stack = torch.stack([F.cross_entropy(lg, labels, reduction="none") for lg in logits_list])  # (M,B)

        # S_m using aggregated peer losses
        selections: List[torch.Tensor] = []
        for m_idx in range(len(models)):
            agg_loss = aggregate_losses(loss_stack, m_idx, active_mask_tensor, mode=args.aggregation)
            k = max(1, int(math.ceil(remember_rate * batch_size)))
            k = min(k, batch_size)
            selected = torch.topk(agg_loss, k, largest=False).indices
            selections.append(selected)

        # selection overlap (pairwise)
        active_pairs = list(combinations([i for i, a in enumerate(active_mask) if a], 2))
        current_overlap = 0.0
        for a_idx, b_idx in active_pairs:
            mask_a = torch.zeros(batch_size, device=images.device, dtype=torch.bool)
            mask_b = torch.zeros(batch_size, device=images.device, dtype=torch.bool)
            mask_a[selections[a_idx]] = True
            mask_b[selections[b_idx]] = True
            overlap = (mask_a & mask_b).sum().float() / mask_a.sum().clamp(min=1)
            batch_accumulator["overlap"].append(overlap.item())
            current_overlap = max(current_overlap, overlap.item())

        # ----------------------------------------------------------------
        # Explore sampling (ablation): when overlap is high, inject high-entropy samples
        # to encourage diversity among models. Disabled by default (--explore_delta=0).
        # Set --explore_delta=0.1 to add 10% high-entropy samples to each model's selection.
        # ----------------------------------------------------------------
        if args.explore_delta > 0 and current_overlap > args.explore_trigger:
            # Compute ensemble entropy to find uncertain samples
            with torch.no_grad():
                probs_ens = torch.stack([lg.softmax(dim=1) for lg in logits_list], dim=0).mean(dim=0)
                entropy = -(probs_ens * (probs_ens + 1e-12).log()).sum(dim=1)  # (B,)
            explore_k = max(1, int(args.explore_delta * batch_size))
            # Select top-entropy samples that are NOT already in any selection
            all_selected = set()
            for sel in selections:
                all_selected.update(sel.cpu().numpy().tolist())
            # Mask already selected
            entropy_masked = entropy.clone()
            for s in all_selected:
                if s < batch_size:
                    entropy_masked[s] = -float('inf')
            explore_candidates = torch.topk(entropy_masked, min(explore_k * 2, batch_size), largest=True).indices
            # Randomly distribute among models to encourage diversity
            np.random.shuffle(explore_candidates.cpu().numpy())
            for m_idx in range(len(models)):
                if m_idx < len(explore_candidates):
                    extra = explore_candidates[m_idx::len(models)][:explore_k // len(models) + 1]
                    if extra.numel() > 0:
                        selections[m_idx] = torch.cat([selections[m_idx], extra.to(selections[m_idx].device)])

        # E-step: compute q (posterior or loss-based)
        # Ablation: set --q_mode=loss to use loss-based sigmoid instead of posterior.
        temp_q = linear_anneal(args.q_temp_max, args.q_temp_min, epoch, args.q_temp_warmup)
        if batch_accumulator["overlap"] and batch_accumulator["overlap"][-1] > args.q_overlap_threshold:
            temp_q = temp_q * (1.0 + args.q_overlap_boost)

        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(models)))
        logits_active = [logits_list[i] for i in active_indices]
        probs_active = [logits.softmax(dim=1) for logits in logits_active]
        p_y_list = [p.gather(1, labels.view(-1, 1)).squeeze(1) for p in probs_active]
        p_ens_y = torch.stack(p_y_list, dim=0).mean(dim=0)

        if args.q_mode == "posterior":
            # Posterior-q (recommended for EM narrative)
            pi_tensor = torch.tensor(pi_t, device=images.device, dtype=p_ens_y.dtype)
            denom = pi_tensor * p_ens_y + (1.0 - pi_tensor) / float(num_classes)
            q_batch = (pi_tensor * p_ens_y) / (denom + 1e-12)
        elif args.q_mode == "bmm":
            # BMM-q: Loss-based score + Beta Mixture Model posterior
            # During warmup, use simple loss-based q
            agg_loss_all = loss_stack[active_mask_tensor].mean(dim=0) if active_mask_tensor.any() else loss_stack.mean(dim=0)
            
            # Accumulate losses for epoch-level BMM fitting
            epoch_losses.append(agg_loss_all.detach().cpu().numpy())
            epoch_indices.append(indices.detach().cpu().numpy())
            
            if epoch < args.bmm_warmup or bmm is None or not bmm.fitted:
                # Warmup: use simple loss-based q
                if args.q_loss_tau == "mean":
                    tau = agg_loss_all.mean()
                else:
                    tau = agg_loss_all.median()
                q_batch = torch.sigmoid((tau - agg_loss_all) / max(temp_q, 1e-6))
            else:
                # Post-warmup: use BMM posterior
                loss_np = agg_loss_all.detach().cpu().numpy()
                scores = loss_to_score(loss_np, outlier_percentile=1.0)
                posteriors = bmm.posterior(scores)
                q_batch = torch.tensor(posteriors, device=images.device, dtype=torch.float32)
        else:
            # Loss-q (engineering-friendly ablation)
            agg_loss_all = loss_stack[active_mask_tensor].mean(dim=0) if active_mask_tensor.any() else loss_stack.mean(dim=0)
            if args.q_loss_tau == "mean":
                tau = agg_loss_all.mean()
            else:
                tau = agg_loss_all.median()
            q_batch = torch.sigmoid((tau - agg_loss_all) / max(temp_q, 1e-6))

        batch_accumulator["q_mean"].append(q_batch.mean().item())
        batch_accumulator["q_std"].append(q_batch.std().item())

        # update global Q (EMA) - update BEFORE using Q_i for weights
        idx_cpu = indices.detach().cpu().numpy().astype(np.int64)
        q_cpu = q_batch.detach().cpu().numpy()
        # Clamp indices to valid range (safety for replay samples)
        valid_mask = idx_cpu < len(q_global)
        idx_cpu = idx_cpu[valid_mask]
        q_cpu = q_cpu[valid_mask]
        q_global[idx_cpu] = args.q_ema * q_global[idx_cpu] + (1.0 - args.q_ema) * q_cpu

        # update pi_t (Beta posterior + EMA)
        sum_q = float(q_batch.sum().item())
        a = args.pi_beta_a
        b = args.pi_beta_b
        pi_hat = (a + sum_q) / (a + b + float(batch_size))
        pi_t = args.pi_ema * pi_t + (1.0 - args.pi_ema) * pi_hat

        # update replay buffer (supports both legacy and purified modes)
        if args.replay_size > 0:
            if args.replay_mode == "purified" and purified_replay is not None:
                # Purified mode: update with BMM posteriors (q_batch)
                # Note: in BMM mode, q_batch already represents clean probability
                purified_replay.update(idx_cpu, q_cpu, current_epoch=epoch)
            else:
                # Legacy mode: simple threshold-based admission with random eviction
                for i, qv in zip(idx_cpu, q_cpu):
                    if qv < args.replay_tau:
                        continue
                    if i in replay_set:
                        continue
                    if len(replay_buffer) < args.replay_size:
                        replay_buffer.append(int(i))
                        replay_set.add(int(i))
                    else:
                        replace_pos = np.random.randint(0, len(replay_buffer))
                        old = replay_buffer[replace_pos]
                        replay_set.discard(old)
                        replay_buffer[replace_pos] = int(i)
                        replay_set.add(int(i))

        # SAM updates per model using peers' selected data
        # Retrieve global Q for current batch (EMA-smoothed responsibilities)
        # Clamp indices and handle out-of-range for safety
        q_global_tensor = torch.zeros(batch_size, device=images.device, dtype=torch.float32)
        for local_idx, global_idx in enumerate(idx_cpu):
            if global_idx < len(q_global):
                q_global_tensor[local_idx] = float(q_global[global_idx])
            else:
                q_global_tensor[local_idx] = float(q_batch[local_idx].item())
        Q_i = q_global_tensor
        
        for m_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            sel = selections[m_idx]
            if sel.numel() == 0:
                continue
            
            # Ablation: set --q_gamma=0 to recover pure hard selection (original co-teaching).
            # Formula: w_i = (1 - gamma) * 1[i in S_m] + gamma * Q_i
            weights = args.q_gamma * Q_i.clone()
            weights[sel] += (1.0 - args.q_gamma)
            
            # Soft-absorb: inactive models still train as "students" but with reduced weight
            # This allows potential recovery. Set student_weight=0 for hard prune ablation.
            student_weight = 0.5 if not active_mask[m_idx] else 1.0
            # Ablation: uncomment next line for hard prune (stop updating inactive models)
            # if not active_mask[m_idx]: continue
            
            weights = weights * student_weight
            
            clean_loss, perturbed_loss = sam_update(
                model, optimizer, images, labels, weights, rho=args.sam_rho
            )
            batch_accumulator["clean_loss"][m_idx] += clean_loss
            batch_accumulator["sharp_loss"][m_idx] += perturbed_loss

        # Track training accuracy on the full batch (pre-update logits)
        for idx, logits in enumerate(logits_list):
            batch_accumulator["train_acc"][idx] += top1_accuracy(logits, labels)

        num_batches += 1
        if (batch_idx + 1) % args.print_freq == 0:
            print(
                f"Epoch [{epoch+1}/{args.n_epoch}] Iter [{batch_idx+1}/{len(loader)}] "
                f"mean_q={batch_accumulator['q_mean'][-1]:.3f} "
                f"std_q={batch_accumulator['q_std'][-1]:.3f}"
            )

    # Reduce metrics
    metrics: Dict[str, float] = {}
    for m_idx in range(len(models)):
        metrics[f"train_acc_{m_idx}"] = batch_accumulator["train_acc"][m_idx] / max(1, num_batches)
        metrics[f"clean_loss_{m_idx}"] = batch_accumulator["clean_loss"][m_idx] / max(1, num_batches)
        metrics[f"sharp_loss_{m_idx}"] = batch_accumulator["sharp_loss"][m_idx] / max(1, num_batches)
    metrics["q_mean"] = float(np.mean(batch_accumulator["q_mean"])) if batch_accumulator["q_mean"] else 0.0
    metrics["q_std"] = float(np.mean(batch_accumulator["q_std"])) if batch_accumulator["q_std"] else 0.0
    metrics["overlap"] = float(np.mean(batch_accumulator["overlap"])) if batch_accumulator["overlap"] else 0.0
    
    # End-of-epoch BMM fitting using accumulated losses
    if args.q_mode == "bmm" and epoch_losses:
        all_losses = np.concatenate(epoch_losses)
        all_indices = np.concatenate(epoch_indices)
        scores = loss_to_score(all_losses, outlier_percentile=1.0)
        # Fit BMM on epoch data
        if bmm is not None:
            try:
                bmm.fit(scores, warm_start=(epoch > args.bmm_warmup))
                metrics["bmm_fitted"] = 1.0
                metrics["bmm_weight_clean"] = float(bmm.weights[1])
            except Exception as e:
                print(f"Warning: BMM fitting failed: {e}")
                metrics["bmm_fitted"] = 0.0
    
    # Add purified replay stats to metrics
    if purified_replay is not None:
        replay_stats = purified_replay.get_statistics()
        metrics["replay_size"] = replay_stats["size"]
        metrics["replay_mean_clean_p"] = replay_stats["mean_clean_p"]
        metrics["replay_admissions"] = replay_stats["total_admissions"]
        metrics["replay_evictions"] = replay_stats["total_evictions"]
    
    return metrics, q_global, pi_t, replay_buffer, replay_set, bmm, purified_replay


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_models < 2:
        raise ValueError("num_models must be at least 2 for interactive co-teaching.")

    set_seed(args.seed)

    # Hyper parameters
    batch_size = args.batch_size
    learning_rate = args.lr

    # load dataset
    input_channel, num_classes, base_train_dataset, test_dataset = load_datasets(args)
    train_dataset, val_dataset = split_train_val(base_train_dataset, args.val_split, args.seed)

    forget_rate = args.noise_rate if args.forget_rate is None else args.forget_rate
    rate_schedule = compute_rate_schedule(
        forget_rate=forget_rate, num_gradual=args.num_gradual, exponent=args.exponent, n_epoch=args.n_epoch
    )

    # Adjust learning rate and betas for Adam Optimizer
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [learning_rate] * args.n_epoch
    beta1_plan = [mom1] * args.n_epoch
    for i in range(args.epoch_decay_start, args.n_epoch):
        alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
        beta1_plan[i] = mom2

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        shuffle=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False,
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )

    # Define models and optimizers
    print("building models...")
    models = []
    optimizers = []
    for m_idx in range(args.num_models):
        net = CNN(input_channel=input_channel, n_outputs=num_classes)
        net.cuda()
        models.append(net)
        optimizers.append(torch.optim.Adam(net.parameters(), lr=learning_rate))

    # Prepare logging
    save_dir = os.path.join(args.result_dir, args.dataset, "srit")
    ensure_dir(save_dir)
    model_str = f"{args.dataset}_srit_{args.noise_type}_{args.noise_rate}"
    txtfile = os.path.join(save_dir, f"{model_str}.txt")
    jsonfile = os.path.join(save_dir, f"{model_str}_training_log.json")
    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if os.path.exists(txtfile):
        os.rename(txtfile, txtfile + f".bak-{now_time}")
    if os.path.exists(jsonfile):
        os.rename(jsonfile, jsonfile + f".bak-{now_time}")
    train_cols = ",".join([f"train_acc_m{i}" for i in range(args.num_models)])
    test_cols = ",".join([f"test_acc_m{i}" for i in range(args.num_models)])
    lambda_cols = ",".join([f"lambda_m{i}" for i in range(args.num_models)])
    header = f"epoch,{train_cols},{test_cols},ensemble_acc,q_mean,q_std,overlap,pi_t,active_count,{lambda_cols}\n"
    with open(txtfile, "w") as f:
        f.write(header)
    
    # Initialize JSON training log for visualization
    training_log = {
        "config": vars(args),
        "epochs": [],
        "metadata": {
            "dataset": args.dataset,
            "noise_type": args.noise_type,
            "noise_rate": args.noise_rate,
            "num_models": args.num_models,
            "start_time": now_time,
        }
    }

    reliability = [1.0 for _ in range(args.num_models)]
    active_mask = [True for _ in range(args.num_models)]
    bad_counts = [0 for _ in range(args.num_models)]
    replay_buffer: List[int] = []
    replay_set: set = set()
    q_global_size = len(base_train_dataset)
    q_global = np.zeros(q_global_size, dtype=np.float32)
    pi_t = float(args.pi_init)
    
    # Initialize BMM and purified replay (will be created in first train_epoch if needed)
    bmm: BetaMixture1D = None
    purified_replay: PurifiedReplayBuffer = None

    # initial evaluation
    test_accs, ensemble_acc = evaluate_models(models, test_loader)
    with open(txtfile, "a") as f:
        row = (
            f"0,"
            + ",".join([f"{0.0:.4f}" for _ in range(args.num_models)])
            + ","
            + ",".join([f"{acc:.4f}" for acc in test_accs])
            + f",{ensemble_acc:.4f},0,0,0,{pi_t:.4f},{sum(active_mask)},"
            + ",".join([f"{lam:.3f}" for lam in reliability])
            + "\n"
        )
        f.write(row)
    print(
        f"Epoch [0/{args.n_epoch}] "
        + " ".join([f"TestAcc_M{i}:{acc:.2f}%" for i, acc in enumerate(test_accs)])
        + f" Ensemble:{ensemble_acc:.2f}%"
    )

    # training
    for epoch in range(1, args.n_epoch):
        remember_rate = 1.0 - rate_schedule[epoch]
        for opt in optimizers:
            adjust_learning_rate(opt, alpha_plan, beta1_plan, epoch)
        train_metrics, q_global, pi_t, replay_buffer, replay_set, bmm, purified_replay = train_epoch(
            epoch,
            args,
            models,
            optimizers,
            train_loader,
            train_dataset,
            num_classes,
            remember_rate,
            reliability,
            active_mask,
            q_global,
            pi_t,
            replay_buffer,
            replay_set,
            bmm,
            purified_replay,
        )
        test_accs, ensemble_acc = evaluate_models(models, test_loader)
        if val_loader is not None:
            val_accs, _ = evaluate_models(models, val_loader)
            reliability = update_reliabilities(
                reliability,
                val_accs,
                decay=args.reliability_decay,
                gap=args.reliability_gap,
                min_lambda=args.reliability_min,
            )
        else:
            train_proxy = [train_metrics.get(f"train_acc_{i}", 0.0) for i in range(args.num_models)]
            reliability = update_reliabilities(
                reliability,
                train_proxy,
                decay=args.reliability_decay,
                gap=args.reliability_gap,
                min_lambda=args.reliability_min,
            )

        # update active mask (soft absorb with patience)
        for i in range(args.num_models):
            if reliability[i] < args.lambda_active:
                bad_counts[i] += 1
            else:
                bad_counts[i] = 0
        for i in range(args.num_models):
            if not active_mask[i]:
                continue
            if bad_counts[i] >= args.lambda_patience:
                if sum(active_mask) > args.min_active:
                    active_mask[i] = False
        print(
                        f"Epoch [{epoch}/{args.n_epoch}] "
            + " ".join([f"TrainAcc_M{i}:{train_metrics[f'train_acc_{i}']:.2f}%" for i in range(args.num_models)])
            + " "
            + " ".join([f"TestAcc_M{i}:{acc:.2f}%" for i, acc in enumerate(test_accs)])
                        + f" Ensemble:{ensemble_acc:.2f}% q_mean:{train_metrics['q_mean']:.3f} "
                            f"q_std:{train_metrics['q_std']:.3f} overlap:{train_metrics['overlap']:.3f} pi:{pi_t:.3f}"
        )
        
        # Save to CSV
        with open(txtfile, "a") as f:
            row = (
                f"{epoch},"
                + ",".join([f"{train_metrics[f'train_acc_{i}']:.4f}" for i in range(args.num_models)])
                + ","
                + ",".join([f"{acc:.4f}" for acc in test_accs])
                                + f",{ensemble_acc:.4f},{train_metrics['q_mean']:.4f},{train_metrics['q_std']:.4f},{train_metrics['overlap']:.4f},{pi_t:.4f},{sum(active_mask)},"
                + ",".join([f"{lam:.3f}" for lam in reliability])
                + "\n"
            )
            f.write(row)
        
        # Save to JSON for visualization
        replay_size_for_log = len(purified_replay) if purified_replay is not None else len(replay_buffer)
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics.get("train_loss", 0.0),
            "train_acc": sum([train_metrics[f'train_acc_{i}'] for i in range(args.num_models)]) / args.num_models,
            "test_acc": ensemble_acc,
            "val_acc": val_accs[0] if val_loader is not None else None,
            "q_mean": train_metrics['q_mean'],
            "q_std": train_metrics['q_std'],
            "overlap": train_metrics['overlap'],
            "pi_t": pi_t,
            "active_models": sum(active_mask),
            "replay_size": replay_size_for_log,
            "test_accs_per_model": test_accs,
            "reliability": reliability,
            # BMM and purified replay metrics
            "bmm_fitted": train_metrics.get("bmm_fitted", 0.0),
            "bmm_weight_clean": train_metrics.get("bmm_weight_clean", 0.0),
            "replay_mean_clean_p": train_metrics.get("replay_mean_clean_p", 0.0),
            "replay_admissions": train_metrics.get("replay_admissions", 0),
            "replay_evictions": train_metrics.get("replay_evictions", 0),
        }
        training_log["epochs"].append(epoch_log)
        
        # Save JSON periodically (every epoch for recovery)
        with open(jsonfile, 'w') as f:
            json.dump(training_log, f, indent=2)


if __name__ == "__main__":
    main()
