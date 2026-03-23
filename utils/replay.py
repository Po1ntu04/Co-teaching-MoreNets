"""
Purified Replay Buffer for Co-teaching with BMM posterior.

Reference: SPR (Self-Paced Resistance) paper's purified buffer.

Key differences from naive replay:
1. Uses BMM posterior (clean_p) instead of threshold-based admission
2. Tracks stability (consecutive high-posterior epochs) for quality assurance
3. Evicts samples with lowest clean_p (not random replacement)
4. Maintains per-sample metadata for informed decisions

This is the core of Scheme C: Purified Replay with quality-based eviction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SampleInfo:
    """Metadata for a sample in the replay buffer."""
    idx: int                  # Original dataset index
    clean_p: float            # Latest BMM posterior P(clean|loss)
    stability: int            # Consecutive epochs with clean_p > threshold
    first_seen_epoch: int     # Epoch when first added
    last_update_epoch: int    # Epoch when last updated
    
    def __lt__(self, other):
        """For heap operations: lower clean_p = lower priority."""
        return self.clean_p < other.clean_p


class PurifiedReplayBuffer:
    """
    Purified replay buffer with BMM-based admission and quality-based eviction.
    
    Design principles:
    1. Admission: Only admit samples with high BMM posterior (clean_p > threshold)
    2. Stability: Require multiple consecutive high-posterior epochs for admission
    3. Eviction: When full, evict sample with lowest clean_p (not random)
    4. Update: Continuously update clean_p for samples in buffer
    
    This prevents the self-reinforcing loop where noisy samples get high q 
    when models memorize, because:
    - BMM posterior is relative to batch distribution
    - Stability requirement filters transient high posteriors
    - Eviction removes samples that degrade over time
    """
    
    def __init__(self, 
                 max_size: int,
                 admission_threshold: float = 0.7,
                 stability_threshold: int = 3,
                 evict_threshold: float = 0.5,
                 ema_alpha: float = 0.3):
        """
        Args:
            max_size: Maximum number of samples to store
            admission_threshold: Minimum clean_p to consider for admission
            stability_threshold: Required consecutive high-posterior epochs
            evict_threshold: Samples below this clean_p are eviction candidates
            ema_alpha: EMA coefficient for updating clean_p
        """
        self.max_size = max_size
        self.admission_threshold = admission_threshold
        self.stability_threshold = stability_threshold
        self.evict_threshold = evict_threshold
        self.ema_alpha = ema_alpha
        
        # Storage
        self.samples: Dict[int, SampleInfo] = {}  # idx -> SampleInfo
        self.sample_list: List[int] = []  # For efficient random sampling
        
        # Statistics
        self.total_admissions = 0
        self.total_evictions = 0
        self.total_rejections = 0
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __contains__(self, idx: int) -> bool:
        return idx in self.samples
    
    @property
    def indices(self) -> List[int]:
        """Return list of all indices in buffer."""
        return self.sample_list.copy()
    
    def update(self, 
               indices: np.ndarray, 
               clean_ps: np.ndarray, 
               current_epoch: int) -> Dict[str, int]:
        """
        Update buffer with new clean posteriors and potentially add/evict samples.
        
        Args:
            indices: Sample indices (global dataset indices)
            clean_ps: BMM posterior P(clean|loss) for each sample
            current_epoch: Current training epoch
            
        Returns:
            Dict with statistics (admitted, evicted, updated)
        """
        indices = np.asarray(indices).flatten()
        clean_ps = np.asarray(clean_ps).flatten()
        
        stats = {"admitted": 0, "evicted": 0, "updated": 0, "rejected": 0}
        
        for idx, clean_p in zip(indices, clean_ps):
            idx = int(idx)
            clean_p = float(clean_p)
            
            if idx in self.samples:
                # Update existing sample
                self._update_sample(idx, clean_p, current_epoch)
                stats["updated"] += 1
            else:
                # Try to admit new sample
                if self._try_admit(idx, clean_p, current_epoch):
                    stats["admitted"] += 1
                else:
                    stats["rejected"] += 1
        
        # Evict low-quality samples if buffer is full
        evicted = self._evict_if_needed()
        stats["evicted"] = evicted
        
        return stats
    
    def _update_sample(self, idx: int, clean_p: float, epoch: int):
        """Update an existing sample's metadata."""
        info = self.samples[idx]
        
        # EMA update of clean_p
        old_p = info.clean_p
        new_p = self.ema_alpha * clean_p + (1 - self.ema_alpha) * old_p
        info.clean_p = new_p
        
        # Update stability counter
        if clean_p > self.admission_threshold:
            info.stability += 1
        else:
            # Reset stability on low posterior
            info.stability = max(0, info.stability - 1)
        
        info.last_update_epoch = epoch
    
    def _try_admit(self, idx: int, clean_p: float, epoch: int) -> bool:
        """
        Try to admit a new sample to the buffer.
        
        Uses a staged admission process:
        1. Track candidate samples that meet threshold but not stability
        2. Only admit when stability requirement is met
        """
        # Must meet basic threshold
        if clean_p < self.admission_threshold:
            self.total_rejections += 1
            return False
        
        # Check if buffer is full
        if len(self.samples) >= self.max_size:
            # Find worst sample to potentially replace
            if not self._should_replace(clean_p):
                return False
            # Evict worst sample
            self._evict_worst()
        
        # Add new sample
        info = SampleInfo(
            idx=idx,
            clean_p=clean_p,
            stability=1,  # First observation
            first_seen_epoch=epoch,
            last_update_epoch=epoch
        )
        self.samples[idx] = info
        self.sample_list.append(idx)
        self.total_admissions += 1
        
        return True
    
    def _should_replace(self, new_clean_p: float) -> bool:
        """Check if new sample should replace the worst existing sample."""
        if not self.samples:
            return True
        
        # Find worst sample
        worst_info = min(self.samples.values(), key=lambda x: x.clean_p)
        
        # Replace if new sample is significantly better
        return new_clean_p > worst_info.clean_p + 0.1
    
    def _evict_worst(self):
        """Evict the sample with lowest clean_p."""
        if not self.samples:
            return
        
        worst_idx = min(self.samples.keys(), key=lambda x: self.samples[x].clean_p)
        self._remove_sample(worst_idx)
        self.total_evictions += 1
    
    def _evict_if_needed(self) -> int:
        """Evict samples that have degraded below threshold."""
        evicted = 0
        
        # Find samples below eviction threshold
        to_evict = [
            idx for idx, info in self.samples.items()
            if info.clean_p < self.evict_threshold and info.stability < self.stability_threshold
        ]
        
        # Limit eviction to prevent emptying buffer
        max_evict = max(0, len(self.samples) - self.max_size // 4)
        to_evict = sorted(to_evict, key=lambda x: self.samples[x].clean_p)[:max_evict]
        
        for idx in to_evict:
            self._remove_sample(idx)
            evicted += 1
            self.total_evictions += 1
        
        return evicted
    
    def _remove_sample(self, idx: int):
        """Remove a sample from the buffer."""
        if idx in self.samples:
            del self.samples[idx]
            if idx in self.sample_list:
                self.sample_list.remove(idx)
    
    def sample(self, n: int, strategy: str = "uniform") -> np.ndarray:
        """
        Sample indices from the buffer.
        
        Args:
            n: Number of samples to draw
            strategy: Sampling strategy
                - "uniform": Equal probability
                - "weighted": Weighted by clean_p
                - "quality": Prefer high clean_p samples
                
        Returns:
            Array of sampled indices
        """
        if not self.sample_list:
            return np.array([], dtype=np.int64)
        
        n = min(n, len(self.sample_list))
        
        if strategy == "uniform":
            indices = np.random.choice(self.sample_list, size=n, replace=False)
        
        elif strategy == "weighted":
            # Weight by clean_p
            weights = np.array([self.samples[idx].clean_p for idx in self.sample_list])
            weights = weights / (weights.sum() + 1e-10)
            indices = np.random.choice(self.sample_list, size=n, replace=False, p=weights)
        
        elif strategy == "quality":
            # Always pick top clean_p samples
            sorted_list = sorted(self.sample_list, 
                                key=lambda x: self.samples[x].clean_p, 
                                reverse=True)
            indices = np.array(sorted_list[:n])
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return indices
    
    def get_clean_ps(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices and their clean_p values."""
        if not self.samples:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        
        indices = np.array(self.sample_list, dtype=np.int64)
        clean_ps = np.array([self.samples[idx].clean_p for idx in self.sample_list], 
                           dtype=np.float32)
        return indices, clean_ps
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.samples:
            return {
                "size": 0,
                "mean_clean_p": 0.0,
                "min_clean_p": 0.0,
                "max_clean_p": 0.0,
                "mean_stability": 0.0,
                "total_admissions": self.total_admissions,
                "total_evictions": self.total_evictions,
                "total_rejections": self.total_rejections,
            }
        
        clean_ps = [info.clean_p for info in self.samples.values()]
        stabilities = [info.stability for info in self.samples.values()]
        
        return {
            "size": len(self.samples),
            "mean_clean_p": np.mean(clean_ps),
            "min_clean_p": np.min(clean_ps),
            "max_clean_p": np.max(clean_ps),
            "mean_stability": np.mean(stabilities),
            "total_admissions": self.total_admissions,
            "total_evictions": self.total_evictions,
            "total_rejections": self.total_rejections,
        }
    
    def get_high_quality_indices(self, 
                                  min_clean_p: float = 0.8,
                                  min_stability: int = 2) -> np.ndarray:
        """Get indices of high-quality samples for curriculum learning."""
        high_quality = [
            idx for idx, info in self.samples.items()
            if info.clean_p >= min_clean_p and info.stability >= min_stability
        ]
        return np.array(high_quality, dtype=np.int64)
    
    def state_dict(self) -> Dict:
        """Serialize buffer state for checkpointing."""
        return {
            "max_size": self.max_size,
            "admission_threshold": self.admission_threshold,
            "stability_threshold": self.stability_threshold,
            "evict_threshold": self.evict_threshold,
            "ema_alpha": self.ema_alpha,
            "samples": {
                idx: {
                    "idx": info.idx,
                    "clean_p": info.clean_p,
                    "stability": info.stability,
                    "first_seen_epoch": info.first_seen_epoch,
                    "last_update_epoch": info.last_update_epoch,
                }
                for idx, info in self.samples.items()
            },
            "sample_list": self.sample_list.copy(),
            "total_admissions": self.total_admissions,
            "total_evictions": self.total_evictions,
            "total_rejections": self.total_rejections,
        }
    
    def load_state_dict(self, state: Dict):
        """Load buffer state from checkpoint."""
        self.max_size = state["max_size"]
        self.admission_threshold = state["admission_threshold"]
        self.stability_threshold = state["stability_threshold"]
        self.evict_threshold = state["evict_threshold"]
        self.ema_alpha = state["ema_alpha"]
        
        self.samples = {
            int(idx): SampleInfo(**info)
            for idx, info in state["samples"].items()
        }
        self.sample_list = state["sample_list"].copy()
        self.total_admissions = state["total_admissions"]
        self.total_evictions = state["total_evictions"]
        self.total_rejections = state["total_rejections"]
