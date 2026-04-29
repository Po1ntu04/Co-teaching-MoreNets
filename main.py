# -*- coding:utf-8 -*-
import argparse
import copy
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
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="optimizer weight decay")
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
    parser.add_argument("--prefetch_factor", type=int, default=4, help="prefetch batches per worker when num_workers > 0")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", help="pin host memory for faster H2D copies")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false", help="disable DataLoader pin_memory")
    parser.add_argument(
        "--persistent_workers",
        dest="persistent_workers",
        action="store_true",
        help="keep DataLoader workers alive across epochs",
    )
    parser.add_argument(
        "--no_persistent_workers",
        dest="persistent_workers",
        action="store_false",
        help="disable DataLoader persistent_workers",
    )
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", help="enable cudnn benchmark")
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false", help="disable cudnn benchmark")
    parser.add_argument("--tf32", dest="tf32", action="store_true", help="allow TF32 matmul/cudnn on Ampere+ GPUs")
    parser.add_argument("--no_tf32", dest="tf32", action="store_false", help="disable TF32 matmul/cudnn")
    parser.add_argument("--num_iter_per_epoch", type=int, default=400)
    parser.add_argument("--epoch_decay_start", type=int, default=80)
    parser.set_defaults(pin_memory=True, persistent_workers=True, cudnn_benchmark=True, tf32=True)
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
        default="hybrid",
        choices=["hybrid", "posterior", "loss", "bmm"],
        help="q computation: hybrid multi-evidence posterior, mixture posterior, loss-q, or bmm posterior",
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
    parser.add_argument("--q_pred_weight", type=float, default=1.0, help="weight for predictive evidence in hybrid q")
    parser.add_argument("--q_margin_weight", type=float, default=1.0, help="weight for margin evidence in hybrid q")
    parser.add_argument(
        "--q_consistency_weight",
        type=float,
        default=1.0,
        help="weight for teacher consistency evidence in hybrid q",
    )
    parser.add_argument("--q_rank_weight", type=float, default=1.0, help="weight for rank evidence in hybrid q")
    # ------------------------------------------------------------------------
    # Prior / pi_t update (slow variable for streaming)
    parser.add_argument("--pi_init", type=float, default=0.8, help="initial clean prior pi")
    parser.add_argument("--pi_ema", type=float, default=0.99, help="EMA for pi_t")
    parser.add_argument("--pi_beta_a", type=float, default=2.0, help="Beta prior a for pi")
    parser.add_argument("--pi_beta_b", type=float, default=2.0, help="Beta prior b for pi")
    # ------------------------------------------------------------------------
    # EMA teacher / generalized M-step
    parser.add_argument("--teacher_ema", type=float, default=0.999, help="EMA decay for slow teacher models")
    parser.add_argument(
        "--mstep_mode",
        type=str,
        default="robust",
        choices=["hard", "soft", "robust"],
        help="hard: top-k CE, soft: Q-weighted CE, robust: CE + teacher KL",
    )
    parser.add_argument("--supervised_alpha", type=float, default=0.7, help="alpha for hard CE in robust M-step")
    parser.add_argument("--q_weight_min", type=float, default=0.05, help="minimum clipped Q weight")
    parser.add_argument("--q_weight_max", type=float, default=0.95, help="maximum clipped Q weight")
    # ------------------------------------------------------------------------
    # Stage-2 reliability-gated utility weighting. Utility is applied only
    # inside peer-selected samples; it is not a clean posterior.
    parser.add_argument(
        "--utility_mode",
        type=str,
        default="none",
        choices=["none", "sam_gap"],
        help="sample utility weighting inside selected samples",
    )
    parser.add_argument("--utility_strength", type=float, default=1.0, help="blend strength for utility weights")
    parser.add_argument("--utility_temp", type=float, default=1.0, help="temperature for standardized utility gap")
    parser.add_argument("--utility_min", type=float, default=0.2, help="minimum utility multiplier")
    parser.add_argument("--utility_max", type=float, default=2.0, help="maximum utility multiplier")
    # ------------------------------------------------------------------------
    # Replay buffer (stream-like stability)
    parser.add_argument("--replay_size", type=int, default=2000, help="max replay buffer size")
    parser.add_argument("--replay_candidate_size", type=int, default=4000, help="candidate buffer size for purified replay")
    parser.add_argument("--replay_ratio", type=float, default=0.25, help="replay sample ratio per batch")
    parser.add_argument("--replay_tau", type=float, default=0.8, help="Q threshold to push into replay (for legacy mode)")
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="purified",
        choices=["legacy", "purified"],
        help="replay buffer mode: legacy threshold buffer or two-stage purified memory",
    )
    parser.add_argument("--replay_admission", type=float, default=0.7, help="Q threshold for admission into purified memory")
    parser.add_argument("--replay_utility", type=float, default=0.75, help="U threshold for admission into purified memory")
    parser.add_argument("--replay_stability", type=int, default=3, help="required consecutive high-Q updates")
    parser.add_argument("--replay_evict", type=float, default=0.5, help="utility threshold for stale memory eviction")
    parser.add_argument("--replay_ema", type=float, default=0.3, help="EMA alpha for Q updates in replay memory")
    parser.add_argument("--replay_u_ema", type=float, default=0.7, help="EMA alpha for U updates in replay memory")
    parser.add_argument("--replay_age_penalty", type=float, default=0.2, help="age penalty in memory utility")
    parser.add_argument("--replay_coverage_weight", type=float, default=0.5, help="coverage gain weight in memory utility")
    parser.add_argument("--replay_redundancy_weight", type=float, default=0.5, help="redundancy penalty in memory utility")
    parser.add_argument("--replay_freq_penalty", type=float, default=0.1, help="sampling penalty for frequently replayed items")
    parser.add_argument("--replay_u_temp", type=float, default=0.5, help="temperature for memory utility sigmoid")
    parser.add_argument(
        "--replay_sample_strategy",
        type=str,
        default="weighted",
        choices=["uniform", "weighted", "quality"],
        help="sampling strategy for purified replay",
    )
    # ------------------------------------------------------------------------
    # Reliability / active set (soft absorb, optional prune)
    parser.add_argument(
        "--lambda_mode",
        type=str,
        default="proxy",
        choices=["proxy", "accuracy"],
        help="proxy: online train-time committee proxies, accuracy: legacy accuracy gap rule",
    )
    parser.add_argument("--lambda_ema", type=float, default=0.9, help="EMA for proxy-based lambda updates")
    parser.add_argument("--lambda_sharp_weight", type=float, default=1.0, help="weight for sharpness gap penalty")
    parser.add_argument(
        "--lambda_disagreement_weight",
        type=float,
        default=1.0,
        help="weight for harmful disagreement penalty",
    )
    parser.add_argument("--lambda_stability_weight", type=float, default=1.0, help="weight for consistency reward")
    parser.add_argument("--lambda_memory_weight", type=float, default=0.5, help="weight for replay alignment reward")
    parser.add_argument("--lambda_active", type=float, default=0.2, help="lambda threshold for active models")
    parser.add_argument("--lambda_patience", type=int, default=5, help="patience before deactivating model")
    parser.add_argument("--min_active", type=int, default=2, help="minimum active models to keep")
    parser.add_argument("--val_split", type=float, default=0.1, help="validation split ratio from train set")
    # ------------------------------------------------------------------------
    # Explore sampling (diversity preservation when overlap is high)
    parser.add_argument("--explore_delta", type=float, default=0.0, help="fraction of batch for explore sampling (0=disabled)")
    parser.add_argument("--explore_trigger", type=float, default=0.85, help="overlap threshold to trigger explore sampling")
    # ------------------------------------------------------------------------
    # Stage-1 target-utility diagnostics. These options only add logging and do
    # not change the training update path.
    parser.add_argument("--diag_alignment", action="store_true", help="enable target-alignment diagnostics")
    parser.add_argument("--diag_every_epoch", type=int, default=5, help="run diagnostics every N epochs")
    parser.add_argument("--diag_batches", type=int, default=4, help="number of train batches used per diagnostic")
    parser.add_argument("--diag_val_batches", type=int, default=2, help="number of validation batches used per diagnostic")
    parser.add_argument(
        "--diag_target",
        type=str,
        default="both",
        choices=["clean", "noisy", "both"],
        help="target labels for diagnostic validation gradients",
    )
    parser.add_argument(
        "--diag_output_dir",
        type=str,
        default="results_diag/stage1_probe",
        help="directory for diagnostic JSON/JSONL outputs",
    )
    # ------------------------------------------------------------------------
    # Stage-2 oracle diagnostics. These options only add logging and do not
    # change model updates. The oracle freezes features and measures the actual
    # validation-loss improvement from a one-step last-layer update per sample.
    parser.add_argument("--diag_oracle", action="store_true", help="enable one-step utility oracle diagnostics")
    parser.add_argument("--diag_oracle_every_epoch", type=int, default=5, help="run oracle diagnostics every N epochs")
    parser.add_argument("--diag_oracle_batches", type=int, default=2, help="number of train batches for oracle diagnostics")
    parser.add_argument("--diag_oracle_val_batches", type=int, default=1, help="number of validation batches for oracle diagnostics")
    parser.add_argument(
        "--diag_oracle_candidates",
        type=int,
        default=128,
        help="max peer-selected candidates per model/batch for oracle scoring",
    )
    parser.add_argument(
        "--diag_oracle_target",
        type=str,
        default="both",
        choices=["clean", "noisy", "both"],
        help="validation target labels used by the oracle",
    )
    parser.add_argument(
        "--diag_oracle_lr",
        type=float,
        default=0.0,
        help="last-layer oracle step size; 0 uses current optimizer lr divided by selected count",
    )
    parser.add_argument(
        "--diag_oracle_output_dir",
        type=str,
        default="results_diag/stage2_oracle",
        help="directory for oracle diagnostic JSON/JSONL outputs",
    )
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def compute_rate_schedule(
    forget_rate: float, num_gradual: int, exponent: float, n_epoch: int
) -> np.ndarray:
    schedule = np.ones(n_epoch) * forget_rate
    gradual = min(max(int(num_gradual), 0), int(n_epoch))
    if gradual > 0:
        schedule[:gradual] = np.linspace(0, forget_rate ** exponent, gradual)
    return schedule


def adjust_learning_rate(optimizer: torch.optim.Optimizer, alpha_plan: Sequence[float], beta1_plan: Sequence[float], epoch: int) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha_plan[epoch]
        if "betas" in param_group:
            param_group["betas"] = (beta1_plan[epoch], 0.999)  # Only change beta1 for Adam.


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


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
    return (per_sample * weights.detach()).sum() / (weights.detach().sum() + 1e-12)


def weighted_kl(logits: torch.Tensor, target_probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = F.kl_div(log_probs, target_probs.detach(), reduction="none").sum(dim=1)
    return (per_sample * weights.detach()).sum() / (weights.detach().sum() + 1e-12)


def js_divergence_from_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    m = 0.5 * (p + q)
    js = 0.5 * (
        F.kl_div(p.log(), m, reduction="none").sum(dim=1)
        + F.kl_div(q.log(), m, reduction="none").sum(dim=1)
    )
    normalizer = max(math.log(p.size(1)), 1e-12)
    return js / normalizer


def normalized_rank(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


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


def build_teacher_models(models: List[torch.nn.Module]) -> List[torch.nn.Module]:
    teachers: List[torch.nn.Module] = []
    for model in models:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        teachers.append(teacher)
    return teachers


def update_ema_model(student: torch.nn.Module, teacher: torch.nn.Module, ema: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(ema).add_(student_param.data, alpha=1.0 - ema)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)


def build_base_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probs: torch.Tensor,
    q_values: torch.Tensor,
    hard_selection: torch.Tensor,
    args,
) -> torch.Tensor:
    q_values = q_values.detach()
    hard_selection = hard_selection.float().detach()
    if args.mstep_mode == "hard":
        weights = hard_selection
        return weighted_ce(logits, labels, weights)
    if args.mstep_mode == "soft":
        weights = q_values.clamp(args.q_weight_min, args.q_weight_max)
        return weighted_ce(logits, labels, weights)

    w = q_values.clamp(args.q_weight_min, args.q_weight_max)
    ce_term = F.cross_entropy(logits, labels, reduction="none")
    kl_term = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs.detach(), reduction="none").sum(dim=1)
    loss = args.supervised_alpha * w * ce_term + (1.0 - args.supervised_alpha) * (1.0 - w) * kl_term
    return loss.mean()


def sam_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_builder,
    rho: float,
) -> Tuple[float, float]:
    """
    Two-step SAM update on a dynamically rebuilt loss.
    Returns: (clean_loss, perturbed_loss)
    """
    optimizer.zero_grad()
    clean_loss = loss_builder()
    clean_loss.backward()
    grad_parts = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grad_parts:
        optimizer.zero_grad()
        return clean_loss.item(), clean_loss.item()
    grad_norm = torch.norm(torch.cat(grad_parts), p=2)
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
    perturbed_loss = loss_builder()
    perturbed_loss.backward()
    with torch.no_grad():
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is None:
                continue
            p.sub_(e_w)
    optimizer.step()
    return clean_loss.item(), perturbed_loss.item()


def standard_update(
    optimizer: torch.optim.Optimizer,
    loss_builder,
) -> Tuple[float, float]:
    optimizer.zero_grad()
    loss = loss_builder()
    loss.backward()
    optimizer.step()
    loss_value = loss.item()
    return loss_value, loss_value


def sam_gap_weighted_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    hard_selection: torch.Tensor,
    student_weight: float,
    rho: float,
    args,
) -> Tuple[float, float, float, float, float]:
    """
    SAM update with utility weighting restricted to peer-selected samples.
    The utility signal is the per-sample sharpness gap under the SAM
    perturbation. Smaller gap means a more stable selected sample.
    """
    selected = hard_selection.float().detach()
    denom = selected.sum().clamp(min=1.0)
    optimizer.zero_grad()
    clean_logits = model(images)
    clean_losses = F.cross_entropy(clean_logits, labels, reduction="none")
    clean_loss = student_weight * (clean_losses * selected).sum() / denom
    clean_loss.backward()

    grad_parts = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grad_parts:
        optimizer.zero_grad()
        loss_value = clean_loss.item()
        return loss_value, loss_value, 1.0, 0.0, 0.0

    grad_norm = torch.norm(torch.cat(grad_parts), p=2)
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
    perturbed_logits = model(images)
    perturbed_losses = F.cross_entropy(perturbed_logits, labels, reduction="none")
    gap = (perturbed_losses - clean_losses.detach()).detach()
    selected_mask = selected > 0
    if selected_mask.any():
        selected_gap = gap[selected_mask]
        centered = selected_gap - selected_gap.mean()
        scale_gap = selected_gap.std(unbiased=False).clamp(min=1e-6) * max(args.utility_temp, 1e-6)
        utility_selected = torch.sigmoid(-centered / scale_gap) * 2.0
        utility_selected = utility_selected.clamp(args.utility_min, args.utility_max)
        utility_selected = utility_selected / utility_selected.mean().clamp(min=1e-6)
        if args.utility_strength < 1.0:
            utility_selected = (1.0 - args.utility_strength) + args.utility_strength * utility_selected
        utility = torch.zeros_like(selected)
        utility[selected_mask] = utility_selected
        utility_mean = float(utility_selected.mean().item())
        utility_std = float(utility_selected.std(unbiased=False).item())
        gap_mean = float(selected_gap.mean().item())
    else:
        utility = selected
        utility_mean = 1.0
        utility_std = 0.0
        gap_mean = 0.0

    weights = selected * utility.detach()
    perturbed_loss = student_weight * (perturbed_losses * weights).sum() / (weights.sum().clamp(min=1e-12))
    perturbed_loss.backward()
    with torch.no_grad():
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is None:
                continue
            p.sub_(e_w)
    optimizer.step()
    return clean_loss.item(), perturbed_loss.item(), utility_mean, utility_std, gap_mean


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


def update_reliabilities_accuracy(
    lambdas: List[float], accuracies: List[float], decay: float, gap: float, min_lambda: float
) -> List[float]:
    best = max(accuracies)
    updated = []
    for lam, acc in zip(lambdas, accuracies):
        if best - acc > gap:
            lam = max(min_lambda, lam * decay)
        updated.append(min(1.0, lam))
    return updated


def update_reliabilities_proxy(args, lambdas: List[float], proxy_scores: List[float]) -> List[float]:
    if not proxy_scores:
        return lambdas
    scores = np.asarray(proxy_scores, dtype=np.float32)
    if np.allclose(scores.std(), 0.0):
        normalized = np.zeros_like(scores)
    else:
        normalized = (scores - scores.mean()) / (scores.std() + 1e-6)
    raw = 1.0 / (1.0 + np.exp(-normalized))
    updated = []
    for lam, value in zip(lambdas, raw):
        new_lam = args.lambda_ema * lam + (1.0 - args.lambda_ema) * float(value)
        updated.append(float(np.clip(new_lam, args.reliability_min, 1.0)))
    return updated


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _as_label_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.astype(np.int64)


def get_clean_labels(dataset, indices: np.ndarray) -> np.ndarray:
    if hasattr(dataset, "train_labels"):
        labels = _as_label_array(dataset.train_labels)
        return labels[indices]
    return np.zeros(len(indices), dtype=np.int64)


def get_clean_mask(dataset, indices: np.ndarray, noisy_labels: np.ndarray) -> np.ndarray:
    if hasattr(dataset, "noise_or_not"):
        clean_flags = np.asarray(dataset.noise_or_not).astype(bool)
        return clean_flags[indices]
    return get_clean_labels(dataset, indices) == noisy_labels


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(original_ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _clean_rate(mask: np.ndarray, clean_flags: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.asarray(clean_flags, dtype=bool)[mask].mean())


def _finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty_like(sorted_values, dtype=np.float64)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    original = np.empty_like(ranks)
    original[order] = ranks
    return original


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_pair(x, y)
    if x.size < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def top_fraction_mask(scores: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.zeros(scores.shape[0], dtype=bool)
    valid = np.isfinite(scores)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return mask
    k = max(1, int(math.ceil(valid_count * fraction)))
    valid_indices = np.where(valid)[0]
    order = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    mask[order[:k]] = True
    return mask


def _safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    try:
        return binary_auc(scores, labels)
    except Exception:
        return float("nan")


def last_layer_error(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return probs - F.one_hot(labels, num_classes=logits.size(1)).to(dtype=probs.dtype)


def adam_last_layer_denominators(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = model.l_c1.weight
    bias = model.l_c1.bias
    state_w = optimizer.state.get(weight, {})
    state_b = optimizer.state.get(bias, {}) if bias is not None else {}
    if "exp_avg_sq" in state_w:
        denom_w = state_w["exp_avg_sq"].detach().sqrt().clamp_min(1e-8)
    else:
        denom_w = torch.ones_like(weight)
    if bias is not None and "exp_avg_sq" in state_b:
        denom_b = state_b["exp_avg_sq"].detach().sqrt().clamp_min(1e-8)
    elif bias is not None:
        denom_b = torch.ones_like(bias)
    else:
        denom_b = torch.ones(weight.size(0), device=weight.device, dtype=weight.dtype)
    return denom_w, denom_b


def alignment_scores(
    train_features: torch.Tensor,
    train_errors: torch.Tensor,
    val_features: torch.Tensor,
    val_errors: torch.Tensor,
    denom_w: torch.Tensor,
    denom_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    val_grad_w = torch.einsum("vc,vd->cd", val_errors, val_features) / max(1, val_features.size(0))
    val_grad_b = val_errors.mean(dim=0)
    raw = torch.einsum("bc,bd,cd->b", train_errors, train_features, val_grad_w)
    raw = raw + torch.einsum("bc,c->b", train_errors, val_grad_b)
    adam = torch.einsum("bc,bd,cd->b", train_errors, train_features, val_grad_w / denom_w)
    adam = adam + torch.einsum("bc,c->b", train_errors, val_grad_b / denom_b)
    return raw, adam


def _feature_logits(model: CNN, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    features = model.extract_features(images)
    logits = model.l_c1(features)
    if model.top_bn:
        logits = model.bn_c1(logits)
    return features, logits


def diagnostic_sam_gap_scores(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    hard_selection: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """
    Non-updating SAM gap diagnostic. Returns per-sample
    loss(theta + epsilon_sam) - loss(theta), then restores parameters.
    """
    if rho <= 0:
        return torch.zeros(labels.size(0), device=labels.device, dtype=torch.float32)

    selected = hard_selection.float().detach()
    denom = selected.sum().clamp(min=1.0)
    model.zero_grad()
    clean_logits = model(images)
    clean_losses = F.cross_entropy(clean_logits, labels, reduction="none")
    clean_loss = (clean_losses * selected).sum() / denom
    clean_loss.backward()

    grad_parts = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not grad_parts:
        model.zero_grad()
        return torch.zeros(labels.size(0), device=labels.device, dtype=torch.float32)

    grad_norm = torch.norm(torch.cat(grad_parts), p=2)
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

    with torch.no_grad():
        perturbed_logits = model(images)
        perturbed_losses = F.cross_entropy(perturbed_logits, labels, reduction="none")
        gap = perturbed_losses - clean_losses.detach()
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is not None:
                p.sub_(e_w)
    model.zero_grad()
    return gap.detach()


def last_layer_one_step_improvement(
    model: CNN,
    train_features: torch.Tensor,
    train_errors: torch.Tensor,
    val_features: torch.Tensor,
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    step_size: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Frozen-feature oracle: for each candidate training sample, simulate one
    last-layer SGD step and measure validation CE improvement.

    Positive value means the candidate update lowers validation loss.
    """
    if train_features.numel() == 0:
        return torch.empty(0, device=val_logits.device, dtype=val_logits.dtype)

    base_loss = F.cross_entropy(val_logits, val_labels, reduction="mean").detach()
    improvements = []
    val_labels_repeated_cache = {}
    use_bias = model.l_c1.bias is not None
    for start in range(0, train_features.size(0), chunk_size):
        end = min(start + chunk_size, train_features.size(0))
        feat_chunk = train_features[start:end]
        err_chunk = train_errors[start:end]
        feature_dot = torch.matmul(feat_chunk, val_features.t())
        if use_bias:
            feature_dot = feature_dot + 1.0
        updated_logits = val_logits.unsqueeze(0) - float(step_size) * feature_dot.unsqueeze(2) * err_chunk.unsqueeze(1)
        flat_logits = updated_logits.reshape(-1, updated_logits.size(-1))
        n_candidates = end - start
        if n_candidates not in val_labels_repeated_cache:
            val_labels_repeated_cache[n_candidates] = val_labels.repeat(n_candidates)
        losses = F.cross_entropy(
            flat_logits,
            val_labels_repeated_cache[n_candidates],
            reduction="none",
        ).view(n_candidates, val_labels.size(0)).mean(dim=1)
        improvements.append(base_loss - losses)
    return torch.cat(improvements, dim=0).detach()


def run_alignment_diagnostics(
    epoch: int,
    args,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    train_loader,
    val_loader,
    base_train_dataset,
    num_classes: int,
    remember_rate: float,
    active_mask: List[bool],
) -> Dict:
    if val_loader is None:
        return {}

    ensure_dir(args.diag_output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = [model.training for model in models]
    for model in models:
        model.eval()

    val_images_parts = []
    val_noisy_parts = []
    val_clean_parts = []
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(val_loader):
            if batch_idx >= args.diag_val_batches:
                break
            idx_np = indices.numpy().astype(np.int64)
            val_images_parts.append(images.to(device, non_blocking=True))
            val_noisy_parts.append(labels.to(device, non_blocking=True).long())
            clean_np = get_clean_labels(base_train_dataset, idx_np)
            val_clean_parts.append(torch.tensor(clean_np, device=device, dtype=torch.long))

    if not val_images_parts:
        for model, training in zip(models, was_training):
            model.train(training)
        return {}

    val_images = torch.cat(val_images_parts, dim=0)
    val_noisy_labels = torch.cat(val_noisy_parts, dim=0)
    val_clean_labels = torch.cat(val_clean_parts, dim=0)
    target_labels = {}
    if args.diag_target in ("clean", "both"):
        target_labels["clean"] = val_clean_labels
    if args.diag_target in ("noisy", "both"):
        target_labels["noisy"] = val_noisy_labels

    active_mask_tensor = torch.tensor(active_mask, device=device, dtype=torch.bool)
    target_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for m_idx, model in enumerate(models):
            val_features, val_logits = _feature_logits(model, val_images)
            for target_name, labels in target_labels.items():
                target_cache[(m_idx, target_name)] = (
                    val_features.detach(),
                    last_layer_error(val_logits, labels).detach(),
                )

    records = []
    per_target_values: Dict[str, Dict[str, List[float]]] = {
        target_name: {
            "loss": [],
            "align_raw": [],
            "align_adam": [],
            "clean": [],
            "selected": [],
        }
        for target_name in target_labels
    }

    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(train_loader):
            if batch_idx >= args.diag_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            indices = indices.to(device, non_blocking=True)
            idx_np = indices.detach().cpu().numpy().astype(np.int64)
            noisy_np = labels.detach().cpu().numpy().astype(np.int64)
            clean_np = get_clean_labels(base_train_dataset, idx_np)
            clean_flags = get_clean_mask(base_train_dataset, idx_np, noisy_np)

            logits_list = [model(images) for model in models]
            loss_stack = torch.stack([F.cross_entropy(logits, labels, reduction="none") for logits in logits_list])
            for m_idx, model in enumerate(models):
                agg_loss = aggregate_losses(loss_stack, m_idx, active_mask_tensor, mode=args.aggregation)
                k = max(1, int(math.ceil(remember_rate * labels.size(0))))
                k = min(k, labels.size(0))
                selected = torch.topk(agg_loss, k, largest=False).indices
                selected_mask = torch.zeros(labels.size(0), device=device, dtype=torch.bool)
                selected_mask[selected] = True

                train_features, train_logits = _feature_logits(model, images)
                train_errors = last_layer_error(train_logits, labels)
                denom_w, denom_b = adam_last_layer_denominators(model, optimizers[m_idx])
                for target_name in target_labels:
                    val_features, val_errors = target_cache[(m_idx, target_name)]
                    raw_scores, adam_scores = alignment_scores(
                        train_features,
                        train_errors,
                        val_features,
                        val_errors,
                        denom_w,
                        denom_b,
                    )
                    loss_np = agg_loss.detach().cpu().numpy().astype(float)
                    raw_np = raw_scores.detach().cpu().numpy().astype(float)
                    adam_np = adam_scores.detach().cpu().numpy().astype(float)
                    selected_np = selected_mask.detach().cpu().numpy().astype(bool)

                    store = per_target_values[target_name]
                    store["loss"].extend(loss_np.tolist())
                    store["align_raw"].extend(raw_np.tolist())
                    store["align_adam"].extend(adam_np.tolist())
                    store["clean"].extend(clean_flags.astype(float).tolist())
                    store["selected"].extend(selected_np.astype(float).tolist())

                    for local_idx in range(labels.size(0)):
                        records.append(
                            {
                                "epoch": int(epoch),
                                "batch": int(batch_idx),
                                "model": int(m_idx),
                                "target": target_name,
                                "index": int(idx_np[local_idx]),
                                "noisy_label": int(noisy_np[local_idx]),
                                "clean_label": int(clean_np[local_idx]),
                                "is_clean": bool(clean_flags[local_idx]),
                                "loss": float(loss_np[local_idx]),
                                "small_loss_selected": bool(selected_np[local_idx]),
                                "align_raw": float(raw_np[local_idx]),
                                "align_adam": float(adam_np[local_idx]),
                            }
                        )

    summaries = {}
    for target_name, values in per_target_values.items():
        loss = np.asarray(values["loss"], dtype=np.float64)
        align_raw = np.asarray(values["align_raw"], dtype=np.float64)
        align_adam = np.asarray(values["align_adam"], dtype=np.float64)
        clean = np.asarray(values["clean"], dtype=bool)
        selected = np.asarray(values["selected"], dtype=bool)
        if loss.size == 0:
            continue
        loss_low = loss <= np.quantile(loss, 0.25)
        loss_high = loss >= np.quantile(loss, 0.75)
        align_high = align_adam >= np.quantile(align_adam, 0.75)
        align_low = align_adam <= np.quantile(align_adam, 0.25)
        summaries[target_name] = {
            "epoch": int(epoch),
            "target": target_name,
            "num_records": int(loss.size),
            "auc_loss_clean": binary_auc(-loss, clean),
            "auc_align_raw_clean": binary_auc(align_raw, clean),
            "auc_align_adam_clean": binary_auc(align_adam, clean),
            "selected_clean_rate": _clean_rate(selected, clean),
            "high_align_clean_rate": _clean_rate(align_high, clean),
            "high_loss_high_align_clean_rate": _clean_rate(loss_high & align_high, clean),
            "low_loss_high_align_clean_rate": _clean_rate(loss_low & align_high, clean),
            "low_loss_low_align_clean_rate": _clean_rate(loss_low & align_low, clean),
            "high_loss_low_align_clean_rate": _clean_rate(loss_high & align_low, clean),
            "mean_loss_clean": _safe_mean(loss[clean]),
            "mean_loss_noisy": _safe_mean(loss[~clean]),
            "mean_align_adam_clean": _safe_mean(align_adam[clean]),
            "mean_align_adam_noisy": _safe_mean(align_adam[~clean]),
        }

    summary_path = os.path.join(args.diag_output_dir, "alignment_summary.jsonl")
    samples_path = os.path.join(args.diag_output_dir, f"alignment_epoch_{epoch:04d}.jsonl")
    with open(summary_path, "a") as f:
        for target_name in sorted(summaries):
            f.write(json.dumps(summaries[target_name]) + "\n")
    with open(samples_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    for model, training in zip(models, was_training):
        model.train(training)
    return {"alignment": summaries, "sample_file": samples_path, "summary_file": summary_path}


def run_utility_oracle_diagnostics(
    epoch: int,
    args,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    train_loader,
    val_loader,
    base_train_dataset,
    remember_rate: float,
    active_mask: List[bool],
) -> Dict:
    """
    Stage-2 diagnostic: inside the peer-selected reliable set, compare cheap
    utility proxies against a frozen-feature one-step validation oracle.
    This function never calls optimizer.step().
    """
    if train_loader is None or val_loader is None:
        return {}

    ensure_dir(args.diag_oracle_output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = [model.training for model in models]
    for model in models:
        model.eval()

    val_images_parts = []
    val_noisy_parts = []
    val_clean_parts = []
    with torch.no_grad():
        for batch_idx, (images, labels, indices) in enumerate(val_loader):
            if batch_idx >= args.diag_oracle_val_batches:
                break
            idx_np = indices.numpy().astype(np.int64)
            val_images_parts.append(images.to(device, non_blocking=True))
            val_noisy_parts.append(labels.to(device, non_blocking=True).long())
            clean_np = get_clean_labels(base_train_dataset, idx_np)
            val_clean_parts.append(torch.tensor(clean_np, device=device, dtype=torch.long))

    if not val_images_parts:
        for model, training in zip(models, was_training):
            model.train(training)
        return {}

    val_images = torch.cat(val_images_parts, dim=0)
    val_noisy_labels = torch.cat(val_noisy_parts, dim=0)
    val_clean_labels = torch.cat(val_clean_parts, dim=0)
    target_labels = {}
    if args.diag_oracle_target in ("clean", "both"):
        target_labels["clean"] = val_clean_labels
    if args.diag_oracle_target in ("noisy", "both"):
        target_labels["noisy"] = val_noisy_labels

    active_mask_tensor = torch.tensor(active_mask, device=device, dtype=torch.bool)
    val_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for m_idx, model in enumerate(models):
            val_features, val_logits = _feature_logits(model, val_images)
            val_cache[m_idx] = (val_features.detach(), val_logits.detach())

    records = []
    per_target_values: Dict[str, Dict[str, List[float]]] = {
        target_name: {
            "oracle": [],
            "neg_loss": [],
            "sam_utility": [],
            "align_raw": [],
            "align_adam": [],
            "clean": [],
        }
        for target_name in target_labels
    }

    for batch_idx, (images, labels, indices) in enumerate(train_loader):
        if batch_idx >= args.diag_oracle_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        indices = indices.to(device, non_blocking=True)
        idx_np = indices.detach().cpu().numpy().astype(np.int64)
        noisy_np = labels.detach().cpu().numpy().astype(np.int64)
        clean_np = get_clean_labels(base_train_dataset, idx_np)
        clean_flags = get_clean_mask(base_train_dataset, idx_np, noisy_np)

        with torch.no_grad():
            logits_list = [model(images) for model in models]
            loss_stack = torch.stack([F.cross_entropy(logits, labels, reduction="none") for logits in logits_list])

        for m_idx, model in enumerate(models):
            agg_loss = aggregate_losses(loss_stack, m_idx, active_mask_tensor, mode=args.aggregation)
            k = max(1, int(math.ceil(remember_rate * labels.size(0))))
            k = min(k, labels.size(0))
            selected = torch.topk(agg_loss, k, largest=False).indices
            selected_sorted = selected[torch.argsort(agg_loss[selected])]
            if args.diag_oracle_candidates > 0 and selected_sorted.numel() > args.diag_oracle_candidates:
                pick = torch.linspace(
                    0,
                    selected_sorted.numel() - 1,
                    steps=args.diag_oracle_candidates,
                    device=selected_sorted.device,
                ).round().long()
                selected_sorted = selected_sorted[pick].unique()

            hard_selection = torch.zeros(labels.size(0), device=device, dtype=torch.float32)
            hard_selection[selected] = 1.0
            sam_gap = diagnostic_sam_gap_scores(
                model,
                images,
                labels,
                hard_selection,
                rho=args.sam_rho,
            )

            with torch.no_grad():
                train_features_all, train_logits_all = _feature_logits(model, images)
                train_errors_all = last_layer_error(train_logits_all, labels)
                denom_w, denom_b = adam_last_layer_denominators(model, optimizers[m_idx])
                val_features, val_logits = val_cache[m_idx]
                step_size = args.diag_oracle_lr
                if step_size <= 0:
                    current_lr = float(optimizers[m_idx].param_groups[0].get("lr", args.lr))
                    step_size = current_lr / float(max(1, k))

                cand = selected_sorted
                cand_np = cand.detach().cpu().numpy().astype(np.int64)
                cand_features = train_features_all[cand].detach()
                cand_errors = train_errors_all[cand].detach()
                cand_loss = agg_loss[cand].detach()
                cand_sam_utility = (-sam_gap[cand]).detach()

                for target_name, val_labels in target_labels.items():
                    val_errors = last_layer_error(val_logits, val_labels)
                    raw_all, adam_all = alignment_scores(
                        train_features_all,
                        train_errors_all,
                        val_features,
                        val_errors,
                        denom_w,
                        denom_b,
                    )
                    oracle = last_layer_one_step_improvement(
                        model,
                        cand_features,
                        cand_errors,
                        val_features,
                        val_logits,
                        val_labels,
                        step_size=step_size,
                    )
                    neg_loss = (-cand_loss).detach()
                    align_raw = raw_all[cand].detach()
                    align_adam = adam_all[cand].detach()

                    oracle_np = oracle.detach().cpu().numpy().astype(float)
                    neg_loss_np = neg_loss.detach().cpu().numpy().astype(float)
                    sam_utility_np = cand_sam_utility.detach().cpu().numpy().astype(float)
                    raw_np = align_raw.detach().cpu().numpy().astype(float)
                    adam_np = align_adam.detach().cpu().numpy().astype(float)
                    clean_sel = clean_flags[cand_np].astype(bool)

                    store = per_target_values[target_name]
                    store["oracle"].extend(oracle_np.tolist())
                    store["neg_loss"].extend(neg_loss_np.tolist())
                    store["sam_utility"].extend(sam_utility_np.tolist())
                    store["align_raw"].extend(raw_np.tolist())
                    store["align_adam"].extend(adam_np.tolist())
                    store["clean"].extend(clean_sel.astype(float).tolist())

                    for j, local_idx in enumerate(cand_np):
                        records.append(
                            {
                                "epoch": int(epoch),
                                "batch": int(batch_idx),
                                "model": int(m_idx),
                                "target": target_name,
                                "index": int(idx_np[local_idx]),
                                "local_index": int(local_idx),
                                "noisy_label": int(noisy_np[local_idx]),
                                "clean_label": int(clean_np[local_idx]),
                                "is_clean": bool(clean_flags[local_idx]),
                                "neg_loss": float(neg_loss_np[j]),
                                "sam_utility": float(sam_utility_np[j]),
                                "align_raw": float(raw_np[j]),
                                "align_adam": float(adam_np[j]),
                                "oracle_improvement": float(oracle_np[j]),
                                "oracle_step_size": float(step_size),
                            }
                        )

    summaries = {}
    for target_name, values in per_target_values.items():
        oracle = np.asarray(values["oracle"], dtype=np.float64)
        neg_loss = np.asarray(values["neg_loss"], dtype=np.float64)
        sam_utility = np.asarray(values["sam_utility"], dtype=np.float64)
        align_raw = np.asarray(values["align_raw"], dtype=np.float64)
        align_adam = np.asarray(values["align_adam"], dtype=np.float64)
        clean = np.asarray(values["clean"], dtype=bool)
        if oracle.size == 0:
            continue

        top_oracle = top_fraction_mask(oracle)
        top_loss = top_fraction_mask(neg_loss)
        top_sam = top_fraction_mask(sam_utility)
        top_align_adam = top_fraction_mask(align_adam)
        finite_oracle = np.isfinite(oracle)
        oracle_positive_rate = float(np.mean(oracle[finite_oracle] > 0.0)) if finite_oracle.any() else float("nan")
        summaries[target_name] = {
            "epoch": int(epoch),
            "target": target_name,
            "num_records": int(oracle.size),
            "oracle_mean": _safe_mean(oracle),
            "oracle_std": float(np.nanstd(oracle)),
            "oracle_positive_rate": oracle_positive_rate,
            "oracle_clean_mean": _safe_mean(oracle[clean]),
            "oracle_noisy_mean": _safe_mean(oracle[~clean]),
            "auc_oracle_clean": _safe_auc(oracle, clean),
            "auc_loss_clean": _safe_auc(neg_loss, clean),
            "auc_sam_utility_clean": _safe_auc(sam_utility, clean),
            "auc_align_adam_clean": _safe_auc(align_adam, clean),
            "pearson_loss_oracle": pearson_corr(neg_loss, oracle),
            "spearman_loss_oracle": spearman_corr(neg_loss, oracle),
            "pearson_sam_utility_oracle": pearson_corr(sam_utility, oracle),
            "spearman_sam_utility_oracle": spearman_corr(sam_utility, oracle),
            "pearson_align_raw_oracle": pearson_corr(align_raw, oracle),
            "spearman_align_raw_oracle": spearman_corr(align_raw, oracle),
            "pearson_align_adam_oracle": pearson_corr(align_adam, oracle),
            "spearman_align_adam_oracle": spearman_corr(align_adam, oracle),
            "top25_oracle_mean_by_oracle": _safe_mean(oracle[top_oracle]),
            "top25_oracle_mean_by_loss": _safe_mean(oracle[top_loss]),
            "top25_oracle_mean_by_sam_utility": _safe_mean(oracle[top_sam]),
            "top25_oracle_mean_by_align_adam": _safe_mean(oracle[top_align_adam]),
            "top25_clean_rate_by_oracle": _clean_rate(top_oracle, clean),
            "top25_clean_rate_by_loss": _clean_rate(top_loss, clean),
            "top25_clean_rate_by_sam_utility": _clean_rate(top_sam, clean),
            "top25_clean_rate_by_align_adam": _clean_rate(top_align_adam, clean),
        }

    summary_path = os.path.join(args.diag_oracle_output_dir, "oracle_summary.jsonl")
    samples_path = os.path.join(args.diag_oracle_output_dir, f"oracle_epoch_{epoch:04d}.jsonl")
    with open(summary_path, "a") as f:
        for target_name in sorted(summaries):
            f.write(json.dumps(summaries[target_name]) + "\n")
    with open(samples_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    for model in models:
        model.zero_grad()
    for model, training in zip(models, was_training):
        model.train(training)
    return {"oracle": summaries, "sample_file": samples_path, "summary_file": summary_path}


def load_datasets(args) -> Tuple:
    if args.dataset == "mnist":
        input_channel = 1
        num_classes = 10
        args.top_bn = False
        args.epoch_decay_start = 80
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
    teacher_models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    loader,
    train_dataset,
    num_classes: int,
    remember_rate: float,
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
    for teacher in teacher_models:
        teacher.eval()
    active_mask_tensor = torch.tensor(active_mask, device=device, dtype=torch.bool)

    if bmm is None and args.q_mode == "bmm":
        bmm = BetaMixture1D(max_iters=args.bmm_max_iters)

    if purified_replay is None and args.replay_mode == "purified" and args.replay_size > 0:
        purified_replay = PurifiedReplayBuffer(
            max_size=args.replay_size,
            candidate_size=args.replay_candidate_size,
            admission_threshold=args.replay_admission,
            utility_threshold=args.replay_utility,
            stability_threshold=args.replay_stability,
            evict_threshold=args.replay_evict,
            q_ema=args.replay_ema,
            u_ema=args.replay_u_ema,
            age_penalty=args.replay_age_penalty,
            coverage_weight=args.replay_coverage_weight,
            redundancy_weight=args.replay_redundancy_weight,
            replay_freq_penalty=args.replay_freq_penalty,
            u_temperature=args.replay_u_temp,
        )

    epoch_losses: List[np.ndarray] = []
    batch_accumulator: Dict[str, List[float]] = {
        "train_acc": [0.0 for _ in models],
        "clean_loss": [0.0 for _ in models],
        "sharp_loss": [0.0 for _ in models],
        "sharp_gap": [0.0 for _ in models],
        "disagreement": [0.0 for _ in models],
        "stability": [0.0 for _ in models],
        "memory_alignment": [0.0 for _ in models],
        "utility_weight_mean": [0.0 for _ in models],
        "utility_weight_std": [0.0 for _ in models],
        "utility_gap_mean": [0.0 for _ in models],
        "q_mean": [],
        "q_std": [],
        "overlap": [],
    }
    num_batches = 0

    for batch_idx, (images, labels, indices) in enumerate(loader):
        if batch_idx >= args.num_iter_per_epoch:
            break
        online_batch_size = len(labels)
        replay_count = 0
        if args.replay_mode == "purified" and purified_replay is not None:
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
            if args.replay_ratio > 0 and replay_buffer:
                replay_count = max(0, int(len(labels) * args.replay_ratio))
                replay_count = min(replay_count, len(replay_buffer))
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
        replay_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if replay_count > 0:
            replay_mask[online_batch_size:] = True

        logits_list = [m(images) for m in models]
        with torch.no_grad():
            teacher_logits_list = [teacher(images) for teacher in teacher_models]
        student_probs_list = [logits.softmax(dim=1) for logits in logits_list]
        teacher_probs_list = [logits.softmax(dim=1) for logits in teacher_logits_list]
        loss_stack = torch.stack([F.cross_entropy(lg, labels, reduction="none") for lg in logits_list])  # (M,B)

        active_indices = [i for i, is_active in enumerate(active_mask) if is_active]
        if not active_indices:
            active_indices = list(range(len(models)))
        committee_probs = torch.stack([student_probs_list[i] for i in active_indices], dim=0).mean(dim=0)
        teacher_committee_probs = torch.stack([teacher_probs_list[i] for i in active_indices], dim=0).mean(dim=0)

        selections: List[torch.Tensor] = []
        for m_idx in range(len(models)):
            agg_loss = aggregate_losses(loss_stack, m_idx, active_mask_tensor, mode=args.aggregation)
            k = max(1, int(math.ceil(remember_rate * batch_size)))
            k = min(k, batch_size)
            selected = torch.topk(agg_loss, k, largest=False).indices
            selections.append(selected)

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

        if args.explore_delta > 0 and current_overlap > args.explore_trigger:
            with torch.no_grad():
                entropy = -(committee_probs * (committee_probs + 1e-12).log()).sum(dim=1)
            explore_k = max(1, int(args.explore_delta * batch_size))
            all_selected = set()
            for sel in selections:
                all_selected.update(sel.cpu().numpy().tolist())
            entropy_masked = entropy.clone()
            for s in all_selected:
                if s < batch_size:
                    entropy_masked[s] = -float("inf")
            explore_candidates = torch.topk(entropy_masked, min(explore_k * 2, batch_size), largest=True).indices
            np.random.shuffle(explore_candidates.cpu().numpy())
            for m_idx in range(len(models)):
                if m_idx < len(explore_candidates):
                    extra = explore_candidates[m_idx::len(models)][:explore_k // len(models) + 1]
                    if extra.numel() > 0:
                        selections[m_idx] = torch.unique(
                            torch.cat([selections[m_idx], extra.to(selections[m_idx].device)])
                        )

        temp_q = linear_anneal(args.q_temp_max, args.q_temp_min, epoch, args.q_temp_warmup)
        if batch_accumulator["overlap"] and batch_accumulator["overlap"][-1] > args.q_overlap_threshold:
            temp_q = temp_q * (1.0 + args.q_overlap_boost)

        p_ens_y = committee_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        agg_loss_all = loss_stack[active_mask_tensor].mean(dim=0) if active_mask_tensor.any() else loss_stack.mean(dim=0)
        epoch_losses.append(agg_loss_all.detach().cpu().numpy())

        if args.q_mode == "hybrid":
            pi_tensor = torch.tensor(pi_t, device=images.device, dtype=p_ens_y.dtype)
            prior_logit = torch.log(pi_tensor * p_ens_y + 1e-12) - torch.log(
                (1.0 - pi_tensor) / float(num_classes) + 1e-12
            )
            top2 = torch.topk(committee_probs, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            consistency = 1.0 - js_divergence_from_probs(committee_probs, teacher_committee_probs)
            rank_clean = 1.0 - normalized_rank(agg_loss_all)
            score = (
                args.q_pred_weight * prior_logit
                + args.q_margin_weight * margin
                + args.q_consistency_weight * consistency
                + args.q_rank_weight * rank_clean
            )
            q_batch = torch.sigmoid(score / max(temp_q, 1e-6))
        elif args.q_mode == "posterior":
            pi_tensor = torch.tensor(pi_t, device=images.device, dtype=p_ens_y.dtype)
            denom = pi_tensor * p_ens_y + (1.0 - pi_tensor) / float(num_classes)
            q_batch = (pi_tensor * p_ens_y) / (denom + 1e-12)
        elif args.q_mode == "bmm":
            if epoch < args.bmm_warmup or bmm is None or not bmm.fitted:
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
            if args.q_loss_tau == "mean":
                tau = agg_loss_all.mean()
            else:
                tau = agg_loss_all.median()
            q_batch = torch.sigmoid((tau - agg_loss_all) / max(temp_q, 1e-6))

        batch_accumulator["q_mean"].append(q_batch.mean().item())
        batch_accumulator["q_std"].append(q_batch.std().item())

        idx_cpu = indices.detach().cpu().numpy().astype(np.int64)
        q_cpu = q_batch.detach().cpu().numpy()
        label_cpu = labels.detach().cpu().numpy().astype(np.int64)
        for idx_value, q_value in zip(idx_cpu, q_cpu):
            if 0 <= idx_value < len(q_global):
                q_global[idx_value] = args.q_ema * q_global[idx_value] + (1.0 - args.q_ema) * q_value

        q_slow_tensor = torch.zeros(batch_size, device=images.device, dtype=torch.float32)
        for local_idx, global_idx in enumerate(idx_cpu):
            if 0 <= global_idx < len(q_global):
                q_slow_tensor[local_idx] = float(q_global[global_idx])
            else:
                q_slow_tensor[local_idx] = q_batch[local_idx].detach()
        Q_i = q_slow_tensor

        sum_q = float(Q_i.sum().item())
        a = args.pi_beta_a
        b = args.pi_beta_b
        pi_hat = (a + sum_q) / (a + b + float(batch_size))
        pi_t = args.pi_ema * pi_t + (1.0 - args.pi_ema) * pi_hat

        if args.replay_size > 0:
            if args.replay_mode == "purified" and purified_replay is not None:
                purified_replay.update(idx_cpu, label_cpu, Q_i.detach().cpu().numpy(), current_epoch=epoch)
            else:
                for i, qv in zip(idx_cpu, Q_i.detach().cpu().numpy()):
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

        for m_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
            sel = selections[m_idx]
            if sel.numel() == 0:
                continue

            hard_selection = torch.zeros(batch_size, device=images.device, dtype=torch.float32)
            hard_selection[sel] = 1.0
            training_q = args.q_gamma * Q_i + (1.0 - args.q_gamma) * hard_selection
            student_weight = 0.5 if not active_mask[m_idx] else 1.0
            teacher_probs_for_loss = teacher_committee_probs.detach()

            def loss_builder(model=model, training_q=training_q, hard_selection=hard_selection, student_weight=student_weight):
                logits = model(images)
                return student_weight * build_base_loss(
                    logits,
                    labels,
                    teacher_probs_for_loss,
                    training_q,
                    hard_selection,
                    args,
                )

            if args.utility_mode == "sam_gap" and args.sam_rho > 0 and args.mstep_mode == "hard":
                clean_loss, perturbed_loss, utility_mean, utility_std, utility_gap_mean = sam_gap_weighted_update(
                    model,
                    optimizer,
                    images,
                    labels,
                    hard_selection,
                    student_weight,
                    rho=args.sam_rho,
                    args=args,
                )
                batch_accumulator["utility_weight_mean"][m_idx] += utility_mean
                batch_accumulator["utility_weight_std"][m_idx] += utility_std
                batch_accumulator["utility_gap_mean"][m_idx] += utility_gap_mean
            elif args.sam_rho > 0:
                clean_loss, perturbed_loss = sam_update(
                    model,
                    optimizer,
                    loss_builder,
                    rho=args.sam_rho,
                )
            else:
                clean_loss, perturbed_loss = standard_update(optimizer, loss_builder)
            batch_accumulator["clean_loss"][m_idx] += clean_loss
            batch_accumulator["sharp_loss"][m_idx] += perturbed_loss
            batch_accumulator["sharp_gap"][m_idx] += max(0.0, perturbed_loss - clean_loss)
            update_ema_model(model, teacher_models[m_idx], args.teacher_ema)

        teacher_pred = teacher_committee_probs.argmax(dim=1)
        for idx, probs in enumerate(student_probs_list):
            pred = probs.argmax(dim=1)
            q_norm = Q_i.detach() / (Q_i.detach().sum() + 1e-12)
            harmful_disagreement = ((pred != teacher_pred).float() * q_norm).sum().item()
            consistency_score = (1.0 - js_divergence_from_probs(probs.detach(), teacher_committee_probs)).mean().item()
            if replay_mask.any():
                memory_alignment = (
                    1.0 - js_divergence_from_probs(probs[replay_mask].detach(), teacher_committee_probs[replay_mask])
                ).mean().item()
            else:
                memory_alignment = consistency_score
            batch_accumulator["disagreement"][idx] += harmful_disagreement
            batch_accumulator["stability"][idx] += consistency_score
            batch_accumulator["memory_alignment"][idx] += memory_alignment
            batch_accumulator["train_acc"][idx] += top1_accuracy(logits_list[idx], labels)

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
        metrics[f"sharp_gap_{m_idx}"] = batch_accumulator["sharp_gap"][m_idx] / max(1, num_batches)
        metrics[f"utility_weight_mean_{m_idx}"] = batch_accumulator["utility_weight_mean"][m_idx] / max(1, num_batches)
        metrics[f"utility_weight_std_{m_idx}"] = batch_accumulator["utility_weight_std"][m_idx] / max(1, num_batches)
        metrics[f"utility_gap_mean_{m_idx}"] = batch_accumulator["utility_gap_mean"][m_idx] / max(1, num_batches)
        metrics[f"disagreement_{m_idx}"] = batch_accumulator["disagreement"][m_idx] / max(1, num_batches)
        metrics[f"stability_{m_idx}"] = batch_accumulator["stability"][m_idx] / max(1, num_batches)
        metrics[f"memory_alignment_{m_idx}"] = batch_accumulator["memory_alignment"][m_idx] / max(1, num_batches)
        metrics[f"proxy_score_{m_idx}"] = (
            -args.lambda_sharp_weight * metrics[f"sharp_gap_{m_idx}"]
            - args.lambda_disagreement_weight * metrics[f"disagreement_{m_idx}"]
            + args.lambda_stability_weight * metrics[f"stability_{m_idx}"]
            + args.lambda_memory_weight * metrics[f"memory_alignment_{m_idx}"]
        )
    metrics["q_mean"] = float(np.mean(batch_accumulator["q_mean"])) if batch_accumulator["q_mean"] else 0.0
    metrics["q_std"] = float(np.mean(batch_accumulator["q_std"])) if batch_accumulator["q_std"] else 0.0
    metrics["overlap"] = float(np.mean(batch_accumulator["overlap"])) if batch_accumulator["overlap"] else 0.0

    if args.q_mode == "bmm" and epoch_losses:
        all_losses = np.concatenate(epoch_losses)
        scores = loss_to_score(all_losses, outlier_percentile=1.0)
        if bmm is not None:
            try:
                bmm.fit(scores, warm_start=(epoch > args.bmm_warmup))
                metrics["bmm_fitted"] = 1.0
                metrics["bmm_weight_clean"] = float(bmm.weights[1])
            except Exception as e:
                print(f"Warning: BMM fitting failed: {e}")
                metrics["bmm_fitted"] = 0.0

    if purified_replay is not None:
        replay_stats = purified_replay.get_statistics()
        metrics["replay_size"] = replay_stats["size"]
        metrics["replay_candidates"] = replay_stats["candidate_size"]
        metrics["replay_mean_clean_p"] = replay_stats["mean_clean_p"]
        metrics["replay_mean_u"] = replay_stats["mean_u"]
        metrics["replay_admissions"] = replay_stats["total_admissions"]
        metrics["replay_evictions"] = replay_stats["total_evictions"]

    return metrics, q_global, pi_t, replay_buffer, replay_set, bmm, purified_replay


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_models < 2:
        raise ValueError("num_models must be at least 2 for interactive co-teaching.")

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)

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

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        drop_last=args.drop_last,
        shuffle=True,
        **loader_kwargs,
    )
    diag_train_loader = None
    if args.diag_alignment or args.diag_oracle:
        diag_loader_kwargs = dict(loader_kwargs)
        if args.num_workers > 0:
            diag_loader_kwargs["persistent_workers"] = False
        diag_train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            drop_last=args.drop_last,
            shuffle=False,
            **diag_loader_kwargs,
        )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            drop_last=False,
            shuffle=False,
            **loader_kwargs,
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        drop_last=False,
        shuffle=False,
        **loader_kwargs,
    )

    # Define models and optimizers
    print("building models...")
    models = []
    optimizers = []
    for m_idx in range(args.num_models):
        net = CNN(input_channel=input_channel, n_outputs=num_classes)
        net.cuda()
        models.append(net)
        optimizers.append(build_optimizer(args, net))
    teacher_models = build_teacher_models(models)

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
    header = (
        f"epoch,{train_cols},{test_cols},ensemble_acc,q_mean,q_std,overlap,pi_t,active_count,"
        f"replay_size,replay_mean_u,{lambda_cols}\n"
    )
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
    q_global = np.full(q_global_size, float(args.pi_init), dtype=np.float32)
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
            + f",{ensemble_acc:.4f},0,0,0,{pi_t:.4f},{sum(active_mask)},0,0,"
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
            teacher_models,
            optimizers,
            train_loader,
            train_dataset,
            num_classes,
            remember_rate,
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
        else:
            val_accs = None
        if args.lambda_mode == "proxy":
            proxy_scores = [train_metrics.get(f"proxy_score_{i}", 0.0) for i in range(args.num_models)]
            reliability = update_reliabilities_proxy(args, reliability, proxy_scores)
        elif val_loader is not None and val_accs is not None:
            reliability = update_reliabilities_accuracy(
                reliability,
                val_accs,
                decay=args.reliability_decay,
                gap=args.reliability_gap,
                min_lambda=args.reliability_min,
            )
        else:
            train_proxy = [train_metrics.get(f"train_acc_{i}", 0.0) for i in range(args.num_models)]
            reliability = update_reliabilities_accuracy(
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
        diag_log = {}
        if (
            args.diag_alignment
            and val_loader is not None
            and args.diag_every_epoch > 0
            and epoch % args.diag_every_epoch == 0
        ):
            diag_log = run_alignment_diagnostics(
                epoch,
                args,
                models,
                optimizers,
                diag_train_loader,
                val_loader,
                base_train_dataset,
                num_classes,
                remember_rate,
                active_mask,
            )
            if diag_log:
                print(f"Diagnostics saved: {diag_log.get('summary_file', '')}")
        if (
            args.diag_oracle
            and val_loader is not None
            and args.diag_oracle_every_epoch > 0
            and epoch % args.diag_oracle_every_epoch == 0
        ):
            oracle_log = run_utility_oracle_diagnostics(
                epoch,
                args,
                models,
                optimizers,
                diag_train_loader,
                val_loader,
                base_train_dataset,
                remember_rate,
                active_mask,
            )
            if oracle_log:
                diag_log.update({"utility_oracle": oracle_log})
                print(f"Oracle diagnostics saved: {oracle_log.get('summary_file', '')}")
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
                + f",{ensemble_acc:.4f},{train_metrics['q_mean']:.4f},{train_metrics['q_std']:.4f},{train_metrics['overlap']:.4f},{pi_t:.4f},{sum(active_mask)},{train_metrics.get('replay_size', 0)},{train_metrics.get('replay_mean_u', 0.0):.4f},"
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
            "utility_weight_mean": sum(
                train_metrics.get(f"utility_weight_mean_{i}", 0.0) for i in range(args.num_models)
            ) / args.num_models,
            "utility_weight_std": sum(
                train_metrics.get(f"utility_weight_std_{i}", 0.0) for i in range(args.num_models)
            ) / args.num_models,
            "utility_gap_mean": sum(
                train_metrics.get(f"utility_gap_mean_{i}", 0.0) for i in range(args.num_models)
            ) / args.num_models,
            # BMM and purified replay metrics
            "bmm_fitted": train_metrics.get("bmm_fitted", 0.0),
            "bmm_weight_clean": train_metrics.get("bmm_weight_clean", 0.0),
            "replay_mean_clean_p": train_metrics.get("replay_mean_clean_p", 0.0),
            "replay_mean_u": train_metrics.get("replay_mean_u", 0.0),
            "replay_candidates": train_metrics.get("replay_candidates", 0),
            "replay_admissions": train_metrics.get("replay_admissions", 0),
            "replay_evictions": train_metrics.get("replay_evictions", 0),
            "diagnostics": diag_log,
        }
        training_log["epochs"].append(epoch_log)
        
        # Save JSON periodically (every epoch for recovery)
        with open(jsonfile, 'w') as f:
            json.dump(training_log, f, indent=2)


if __name__ == "__main__":
    main()
