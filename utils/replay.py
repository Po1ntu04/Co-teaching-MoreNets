"""
Two-stage replay buffer with explicit candidate staging and purified memory.

The buffer follows the first-stage target in target_revised.md:
1. Recent samples first enter a short-lived candidate pool D.
2. Samples are admitted into purified memory R only after they remain
   consistently clean enough and useful enough.
3. Memory samples are refreshed and can be evicted when their utility drops.

This avoids the previous contradiction where the code claimed to require
stability before admission but actually admitted a sample immediately on the
first high posterior.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SampleInfo:
    idx: int
    label: int
    q: float
    u: float
    stability: int
    first_seen_epoch: int
    last_update_epoch: int
    replay_freq: int = 0


class PurifiedReplayBuffer:
    """
    Candidate buffer D + purified memory R.

    The API stays compatible with the previous class name, but the internals are
    now aligned with the target:
    - D handles delayed decisions.
    - R stores only samples that pass both clean and memory-worthiness tests.
    """

    def __init__(
        self,
        max_size: int,
        candidate_size: int = None,
        admission_threshold: float = 0.7,
        utility_threshold: float = 0.7,
        stability_threshold: int = 3,
        evict_threshold: float = 0.5,
        q_ema: float = 0.3,
        u_ema: float = 0.7,
        age_penalty: float = 0.2,
        coverage_weight: float = 0.5,
        redundancy_weight: float = 0.5,
        replay_freq_penalty: float = 0.1,
        u_temperature: float = 0.5,
    ):
        self.max_size = int(max_size)
        self.candidate_size = int(candidate_size or max(2 * max_size, 1))
        self.admission_threshold = float(admission_threshold)
        self.utility_threshold = float(utility_threshold)
        self.stability_threshold = int(stability_threshold)
        self.evict_threshold = float(evict_threshold)
        self.q_ema = float(q_ema)
        self.u_ema = float(u_ema)
        self.age_penalty = float(age_penalty)
        self.coverage_weight = float(coverage_weight)
        self.redundancy_weight = float(redundancy_weight)
        self.replay_freq_penalty = float(replay_freq_penalty)
        self.u_temperature = float(u_temperature)

        self.candidates: Dict[int, SampleInfo] = {}
        self.memory: Dict[int, SampleInfo] = {}
        self.memory_list: List[int] = []

        self.total_admissions = 0
        self.total_evictions = 0
        self.total_rejections = 0
        self.total_candidate_replacements = 0

    def __len__(self) -> int:
        return len(self.memory)

    def __contains__(self, idx: int) -> bool:
        return int(idx) in self.memory

    @property
    def indices(self) -> List[int]:
        return self.memory_list.copy()

    def update(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        clean_ps: np.ndarray,
        current_epoch: int,
    ) -> Dict[str, int]:
        indices = np.asarray(indices).flatten()
        labels = np.asarray(labels).flatten()
        clean_ps = np.asarray(clean_ps).flatten()

        stats = {"admitted": 0, "evicted": 0, "updated": 0, "rejected": 0}
        for idx, label, clean_p in zip(indices, labels, clean_ps):
            idx = int(idx)
            label = int(label)
            clean_p = float(clean_p)
            if idx in self.memory:
                self._refresh_info(self.memory[idx], clean_p, label, current_epoch)
                stats["updated"] += 1
                continue

            if idx in self.candidates:
                info = self.candidates[idx]
                self._refresh_info(info, clean_p, label, current_epoch)
                stats["updated"] += 1
            else:
                info = SampleInfo(
                    idx=idx,
                    label=label,
                    q=clean_p,
                    u=0.0,
                    stability=1 if clean_p >= self.admission_threshold else 0,
                    first_seen_epoch=current_epoch,
                    last_update_epoch=current_epoch,
                )
                self.candidates[idx] = info

            self._update_utility(info, current_epoch, in_memory=False)
            if self._eligible_for_admission(info):
                if self._admit(info):
                    stats["admitted"] += 1
                else:
                    stats["rejected"] += 1

        self._refresh_memory(current_epoch)
        stats["evicted"] += self._evict_if_needed(current_epoch)
        stats["rejected"] += self._shrink_candidates()
        return stats

    def _refresh_info(self, info: SampleInfo, clean_p: float, label: int, epoch: int) -> None:
        info.q = self.q_ema * clean_p + (1.0 - self.q_ema) * info.q
        info.label = int(label)
        if clean_p >= self.admission_threshold:
            info.stability += 1
        else:
            info.stability = max(0, info.stability - 1)
        info.last_update_epoch = epoch

    def _eligible_for_admission(self, info: SampleInfo) -> bool:
        return (
            info.q >= self.admission_threshold
            and info.u >= self.utility_threshold
            and info.stability >= self.stability_threshold
        )

    def _admit(self, info: SampleInfo) -> bool:
        if info.idx in self.memory:
            return True
        if info.idx in self.candidates:
            del self.candidates[info.idx]
        self.memory[info.idx] = info
        if info.idx not in self.memory_list:
            self.memory_list.append(info.idx)
        self.total_admissions += 1
        self._evict_if_needed(info.last_update_epoch)
        return info.idx in self.memory

    def _refresh_memory(self, epoch: int) -> None:
        for info in list(self.memory.values()):
            self._update_utility(info, epoch, in_memory=True)

    def _update_utility(self, info: SampleInfo, epoch: int, in_memory: bool) -> None:
        coverage_gain = self._coverage_need(info.label, exclude_idx=info.idx if in_memory else None)
        redundancy = self._redundancy(info.label, exclude_idx=info.idx if in_memory else None)
        age_norm = self._age_norm(info, epoch)
        raw_value = (
            info.q
            + self.coverage_weight * coverage_gain
            - self.redundancy_weight * redundancy
            - self.age_penalty * age_norm
        )
        u_hat = 1.0 / (1.0 + np.exp(-raw_value / max(self.u_temperature, 1e-6)))
        if info.u <= 0.0:
            info.u = float(u_hat)
        else:
            info.u = self.u_ema * info.u + (1.0 - self.u_ema) * float(u_hat)

    def _coverage_need(self, label: int, exclude_idx: int = None) -> float:
        counts = self._memory_label_counts(exclude_idx=exclude_idx)
        if not counts:
            return 1.0
        max_count = max(counts.values())
        label_count = counts.get(int(label), 0)
        if max_count <= 0:
            return 1.0
        return 1.0 - float(label_count) / float(max_count)

    def _redundancy(self, label: int, exclude_idx: int = None) -> float:
        counts = self._memory_label_counts(exclude_idx=exclude_idx)
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        return float(counts.get(int(label), 0)) / float(total)

    def _memory_label_counts(self, exclude_idx: int = None) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for idx, info in self.memory.items():
            if exclude_idx is not None and idx == exclude_idx:
                continue
            counts[info.label] = counts.get(info.label, 0) + 1
        return counts

    def _age_norm(self, info: SampleInfo, epoch: int) -> float:
        age = max(0, int(epoch) - int(info.first_seen_epoch))
        denom = max(1.0, 4.0 * float(self.stability_threshold))
        return min(float(age) / denom, 1.0)

    def _evict_score(self, info: SampleInfo, epoch: int) -> float:
        coverage_need = self._coverage_need(info.label, exclude_idx=info.idx)
        redundancy = self._redundancy(info.label, exclude_idx=info.idx)
        age_norm = self._age_norm(info, epoch)
        return (
            (1.0 - info.u)
            + self.redundancy_weight * redundancy
            + self.age_penalty * age_norm
            - self.coverage_weight * coverage_need
        )

    def _evict_if_needed(self, epoch: int) -> int:
        evicted = 0
        while len(self.memory) > self.max_size:
            worst_idx = max(self.memory.keys(), key=lambda idx: self._evict_score(self.memory[idx], epoch))
            self._remove_from_memory(worst_idx)
            evicted += 1

        stale = [
            idx
            for idx, info in self.memory.items()
            if info.u < self.evict_threshold and info.stability < self.stability_threshold
        ]
        for idx in stale:
            if len(self.memory) <= max(1, self.max_size // 4):
                break
            self._remove_from_memory(idx)
            evicted += 1
        self.total_evictions += evicted
        return evicted

    def _remove_from_memory(self, idx: int) -> None:
        if idx in self.memory:
            del self.memory[idx]
        if idx in self.memory_list:
            self.memory_list.remove(idx)

    def _shrink_candidates(self) -> int:
        if len(self.candidates) <= self.candidate_size:
            return 0
        overflow = len(self.candidates) - self.candidate_size
        ordered = sorted(
            self.candidates.values(),
            key=lambda info: (info.last_update_epoch, info.q, info.u),
        )
        for info in ordered[:overflow]:
            self.candidates.pop(info.idx, None)
        self.total_candidate_replacements += overflow
        self.total_rejections += overflow
        return overflow

    def sample(self, n: int, strategy: str = "uniform") -> np.ndarray:
        if not self.memory_list:
            return np.array([], dtype=np.int64)

        n = min(int(n), len(self.memory_list))
        if strategy == "uniform":
            sampled = np.random.choice(self.memory_list, size=n, replace=False)
        elif strategy == "quality":
            sampled = np.array(
                sorted(self.memory_list, key=lambda idx: self.memory[idx].u, reverse=True)[:n],
                dtype=np.int64,
            )
        else:
            weights = np.array([self._sampling_weight(self.memory[idx]) for idx in self.memory_list], dtype=np.float64)
            weights = weights / (weights.sum() + 1e-12)
            sampled = np.random.choice(self.memory_list, size=n, replace=False, p=weights)

        for idx in sampled:
            self.memory[int(idx)].replay_freq += 1
        return np.asarray(sampled, dtype=np.int64)

    def _sampling_weight(self, info: SampleInfo) -> float:
        coverage_need = self._coverage_need(info.label, exclude_idx=info.idx)
        value = (
            info.u
            + self.coverage_weight * coverage_need
            - self.replay_freq_penalty * float(info.replay_freq)
        )
        return float(np.exp(np.clip(value, -10.0, 10.0)))

    def get_clean_ps(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.memory:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        indices = np.array(self.memory_list, dtype=np.int64)
        clean_ps = np.array([self.memory[idx].q for idx in self.memory_list], dtype=np.float32)
        return indices, clean_ps

    def get_high_quality_indices(self, min_clean_p: float = 0.8, min_stability: int = 2) -> np.ndarray:
        indices = [
            idx
            for idx, info in self.memory.items()
            if info.q >= min_clean_p and info.stability >= min_stability
        ]
        return np.array(indices, dtype=np.int64)

    def get_statistics(self) -> Dict[str, float]:
        if not self.memory:
            return {
                "size": 0,
                "candidate_size": len(self.candidates),
                "mean_clean_p": 0.0,
                "mean_u": 0.0,
                "min_clean_p": 0.0,
                "max_clean_p": 0.0,
                "mean_stability": 0.0,
                "coverage": 0.0,
                "total_admissions": self.total_admissions,
                "total_evictions": self.total_evictions,
                "total_rejections": self.total_rejections,
            }
        qs = np.array([info.q for info in self.memory.values()], dtype=np.float32)
        us = np.array([info.u for info in self.memory.values()], dtype=np.float32)
        stabilities = np.array([info.stability for info in self.memory.values()], dtype=np.float32)
        labels = {info.label for info in self.memory.values()}
        return {
            "size": len(self.memory),
            "candidate_size": len(self.candidates),
            "mean_clean_p": float(qs.mean()),
            "mean_u": float(us.mean()),
            "min_clean_p": float(qs.min()),
            "max_clean_p": float(qs.max()),
            "mean_stability": float(stabilities.mean()),
            "coverage": float(len(labels)),
            "total_admissions": self.total_admissions,
            "total_evictions": self.total_evictions,
            "total_rejections": self.total_rejections,
        }

    def state_dict(self) -> Dict:
        return {
            "max_size": self.max_size,
            "candidate_size": self.candidate_size,
            "admission_threshold": self.admission_threshold,
            "utility_threshold": self.utility_threshold,
            "stability_threshold": self.stability_threshold,
            "evict_threshold": self.evict_threshold,
            "q_ema": self.q_ema,
            "u_ema": self.u_ema,
            "age_penalty": self.age_penalty,
            "coverage_weight": self.coverage_weight,
            "redundancy_weight": self.redundancy_weight,
            "replay_freq_penalty": self.replay_freq_penalty,
            "u_temperature": self.u_temperature,
            "candidates": {idx: vars(info) for idx, info in self.candidates.items()},
            "memory": {idx: vars(info) for idx, info in self.memory.items()},
            "memory_list": self.memory_list.copy(),
            "total_admissions": self.total_admissions,
            "total_evictions": self.total_evictions,
            "total_rejections": self.total_rejections,
            "total_candidate_replacements": self.total_candidate_replacements,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.max_size = state["max_size"]
        self.candidate_size = state["candidate_size"]
        self.admission_threshold = state["admission_threshold"]
        self.utility_threshold = state["utility_threshold"]
        self.stability_threshold = state["stability_threshold"]
        self.evict_threshold = state["evict_threshold"]
        self.q_ema = state["q_ema"]
        self.u_ema = state["u_ema"]
        self.age_penalty = state["age_penalty"]
        self.coverage_weight = state["coverage_weight"]
        self.redundancy_weight = state["redundancy_weight"]
        self.replay_freq_penalty = state["replay_freq_penalty"]
        self.u_temperature = state["u_temperature"]
        self.candidates = {int(idx): SampleInfo(**info) for idx, info in state["candidates"].items()}
        self.memory = {int(idx): SampleInfo(**info) for idx, info in state["memory"].items()}
        self.memory_list = state["memory_list"].copy()
        self.total_admissions = state["total_admissions"]
        self.total_evictions = state["total_evictions"]
        self.total_rejections = state["total_rejections"]
        self.total_candidate_replacements = state["total_candidate_replacements"]
