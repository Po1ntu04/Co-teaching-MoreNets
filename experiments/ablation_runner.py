#!/usr/bin/env python3
"""
Ablation Study Runner for Co-teaching EM Framework
===================================================
按照科学实验范式，逐步验证各组件的贡献。

实验设计遵循以下原则：
1. 单变量原则：每次只改变一个因素
2. 基线对照：始终与原始Co-teaching对比
3. 由简入繁：从核心机制到辅助组件

使用方法:
    python experiments/ablation_runner.py --stage 1  # 运行第一阶段
    python experiments/ablation_runner.py --all      # 运行全部
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# 实验结果保存目录
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 实验阶段定义
# =============================================================================

EXPERIMENTS = {
    # =========================================================================
    # Stage 1: 基线验证 - 确认原始Co-teaching在当前代码中能复现
    # =========================================================================
    "stage1_baseline": {
        "name": "Stage 1: Baseline Reproduction",
        "description": "验证原始Co-teaching在不同噪声率下的表现",
        "experiments": [
            {
                "name": "coteaching_sym20",
                "desc": "Co-teaching baseline (symmetric 20%)",
                "args": "--q_gamma 0 --replay_size 0 --explore_delta 0 --noise_rate 0.2 --noise_type symmetric"
            },
            {
                "name": "coteaching_sym50",
                "desc": "Co-teaching baseline (symmetric 50%)",
                "args": "--q_gamma 0 --replay_size 0 --explore_delta 0 --noise_rate 0.5 --noise_type symmetric"
            },
            {
                "name": "coteaching_asym40",
                "desc": "Co-teaching baseline (asymmetric 40%)",
                "args": "--q_gamma 0 --replay_size 0 --explore_delta 0 --noise_rate 0.4 --noise_type asymmetric"
            },
        ]
    },
    
    # =========================================================================
    # Stage 2: Q计算方式消融 - posterior vs loss-based
    # =========================================================================
    "stage2_q_mode": {
        "name": "Stage 2: Q Computation Ablation",
        "description": "比较posterior-q (EM理论) vs loss-q (工程方案)",
        "experiments": [
            {
                "name": "q_posterior_sym50",
                "desc": "Posterior-q (EM narrative)",
                "args": "--q_mode posterior --q_gamma 0.5 --noise_rate 0.5"
            },
            {
                "name": "q_loss_sym50",
                "desc": "Loss-based q (engineering)",
                "args": "--q_mode loss --q_gamma 0.5 --noise_rate 0.5"
            },
            {
                "name": "q_posterior_asym40",
                "desc": "Posterior-q on asymmetric noise",
                "args": "--q_mode posterior --q_gamma 0.5 --noise_rate 0.4 --noise_type asymmetric"
            },
            {
                "name": "q_loss_asym40",
                "desc": "Loss-based q on asymmetric noise",
                "args": "--q_mode loss --q_gamma 0.5 --noise_rate 0.4 --noise_type asymmetric"
            },
        ]
    },
    
    # =========================================================================
    # Stage 3: Q-gamma混合权重 - hard selection vs soft weighting
    # =========================================================================
    "stage3_q_gamma": {
        "name": "Stage 3: Q-Gamma Mixing Ablation",
        "description": "验证hard selection (gamma=0) vs soft weighting (gamma>0)",
        "experiments": [
            {
                "name": "gamma_0",
                "desc": "Pure hard selection (original co-teaching)",
                "args": "--q_gamma 0.0 --noise_rate 0.5"
            },
            {
                "name": "gamma_0.3",
                "desc": "Light soft weighting",
                "args": "--q_gamma 0.3 --noise_rate 0.5"
            },
            {
                "name": "gamma_0.5",
                "desc": "Balanced mixing",
                "args": "--q_gamma 0.5 --noise_rate 0.5"
            },
            {
                "name": "gamma_0.7",
                "desc": "Heavy soft weighting",
                "args": "--q_gamma 0.7 --noise_rate 0.5"
            },
            {
                "name": "gamma_1.0",
                "desc": "Pure Q weighting (no selection)",
                "args": "--q_gamma 1.0 --noise_rate 0.5"
            },
        ]
    },
    
    # =========================================================================
    # Stage 4: Replay Buffer - 流式场景下的记忆机制
    # =========================================================================
    "stage4_replay": {
        "name": "Stage 4: Replay Buffer Ablation",
        "description": "验证replay buffer对流式稳定性的影响",
        "experiments": [
            {
                "name": "replay_off",
                "desc": "No replay (pure streaming)",
                "args": "--replay_size 0 --noise_rate 0.5"
            },
            {
                "name": "replay_1k",
                "desc": "Small replay buffer (1K)",
                "args": "--replay_size 1000 --replay_ratio 0.1 --noise_rate 0.5"
            },
            {
                "name": "replay_5k",
                "desc": "Medium replay buffer (5K)",
                "args": "--replay_size 5000 --replay_ratio 0.2 --noise_rate 0.5"
            },
            {
                "name": "replay_10k",
                "desc": "Large replay buffer (10K)",
                "args": "--replay_size 10000 --replay_ratio 0.3 --noise_rate 0.5"
            },
        ]
    },
    
    # =========================================================================
    # Stage 5: 模型数量 & Soft-absorb机制
    # =========================================================================
    "stage5_ensemble": {
        "name": "Stage 5: Ensemble Size & Absorb Ablation",
        "description": "多模型数量与soft-absorb机制的影响",
        "experiments": [
            {
                "name": "M2_no_absorb",
                "desc": "2 models, no absorb",
                "args": "--n_models 2 --lambda_active 0 --noise_rate 0.5"
            },
            {
                "name": "M3_no_absorb",
                "desc": "3 models, no absorb",
                "args": "--n_models 3 --lambda_active 0 --noise_rate 0.5"
            },
            {
                "name": "M3_soft_absorb",
                "desc": "3 models, soft absorb",
                "args": "--n_models 3 --lambda_active 0.4 --lambda_patience 3 --noise_rate 0.5"
            },
            {
                "name": "M5_soft_absorb",
                "desc": "5 models, soft absorb",
                "args": "--n_models 5 --lambda_active 0.4 --lambda_patience 3 --noise_rate 0.5"
            },
        ]
    },
    
    # =========================================================================
    # Stage 6: 探索采样 - 高重叠时的多样性保持
    # =========================================================================
    "stage6_explore": {
        "name": "Stage 6: Explore Sampling Ablation",
        "description": "高熵样本探索对多样性的影响",
        "experiments": [
            {
                "name": "explore_off",
                "desc": "No explore sampling",
                "args": "--explore_delta 0 --noise_rate 0.5"
            },
            {
                "name": "explore_5pct",
                "desc": "5% explore sampling",
                "args": "--explore_delta 0.05 --explore_trigger 0.8 --noise_rate 0.5"
            },
            {
                "name": "explore_10pct",
                "desc": "10% explore sampling",
                "args": "--explore_delta 0.1 --explore_trigger 0.8 --noise_rate 0.5"
            },
            {
                "name": "explore_adaptive",
                "desc": "10% explore, lower trigger",
                "args": "--explore_delta 0.1 --explore_trigger 0.7 --noise_rate 0.5"
            },
        ]
    },
    
    # =========================================================================
    # Stage 7: 完整系统 vs 消融组合
    # =========================================================================
    "stage7_full_system": {
        "name": "Stage 7: Full System Comparison",
        "description": "完整系统与各消融版本的最终对比",
        "experiments": [
            {
                "name": "baseline_coteaching",
                "desc": "Original Co-teaching (all off)",
                "args": "--q_gamma 0 --replay_size 0 --explore_delta 0 --lambda_active 0 --noise_rate 0.5"
            },
            {
                "name": "full_system",
                "desc": "Full EM system (all on)",
                "args": "--q_mode posterior --q_gamma 0.5 --replay_size 5000 --replay_ratio 0.2 --explore_delta 0.1 --lambda_active 0.4 --noise_rate 0.5"
            },
            {
                "name": "no_replay",
                "desc": "Full - replay",
                "args": "--q_mode posterior --q_gamma 0.5 --replay_size 0 --explore_delta 0.1 --lambda_active 0.4 --noise_rate 0.5"
            },
            {
                "name": "no_explore",
                "desc": "Full - explore",
                "args": "--q_mode posterior --q_gamma 0.5 --replay_size 5000 --explore_delta 0 --lambda_active 0.4 --noise_rate 0.5"
            },
            {
                "name": "no_soft_q",
                "desc": "Full - soft q (hard selection only)",
                "args": "--q_mode posterior --q_gamma 0 --replay_size 5000 --explore_delta 0.1 --lambda_active 0.4 --noise_rate 0.5"
            },
        ]
    },
}

# 通用参数
COMMON_ARGS = "--dataset cifar10 --n_epoch 100 --seed 42 --val_split 0.1"


def run_experiment(name: str, args: str, desc: str, dry_run: bool = False):
    """运行单个实验"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"{name}_{timestamp}.json"
    
    cmd = f"python main.py {COMMON_ARGS} {args} --exp_name {name}"
    
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Description: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # 保存结果
        with open(log_file, 'w') as f:
            json.dump({
                "name": name,
                "desc": desc,
                "args": args,
                "returncode": result.returncode,
                "stdout": result.stdout[-5000:],  # 只保留最后5000字符
                "stderr": result.stderr[-2000:],
                "timestamp": timestamp
            }, f, indent=2)
        
        if result.returncode == 0:
            print(f"✓ Completed. Log saved to {log_file}")
        else:
            print(f"✗ Failed with code {result.returncode}")
            print(result.stderr[-500:])
            
    except Exception as e:
        print(f"✗ Error: {e}")


def run_stage(stage_key: str, dry_run: bool = False):
    """运行指定阶段的所有实验"""
    if stage_key not in EXPERIMENTS:
        print(f"Unknown stage: {stage_key}")
        print(f"Available stages: {list(EXPERIMENTS.keys())}")
        return
    
    stage = EXPERIMENTS[stage_key]
    print(f"\n{'#'*60}")
    print(f"# {stage['name']}")
    print(f"# {stage['description']}")
    print(f"{'#'*60}")
    
    for exp in stage["experiments"]:
        run_experiment(exp["name"], exp["args"], exp["desc"], dry_run)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument("--stage", type=str, help="Stage to run (e.g., stage1_baseline)")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()
    
    if args.list:
        print("\n=== Available Experiment Stages ===\n")
        for key, stage in EXPERIMENTS.items():
            print(f"{key}:")
            print(f"  {stage['name']}")
            print(f"  {stage['description']}")
            for exp in stage['experiments']:
                print(f"    - {exp['name']}: {exp['desc']}")
            print()
        return
    
    if args.all:
        for stage_key in EXPERIMENTS:
            run_stage(stage_key, args.dry_run)
    elif args.stage:
        run_stage(args.stage, args.dry_run)
    else:
        parser.print_help()
        print("\n推荐的实验顺序：")
        for i, key in enumerate(EXPERIMENTS.keys(), 1):
            print(f"  {i}. python experiments/ablation_runner.py --stage {key}")


if __name__ == "__main__":
    main()
