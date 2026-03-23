"""
Beta Mixture Model (BMM) for clean/noisy sample separation.

Reference: SPR (Self-Paced Resistance) paper
Key insight: Use loss-based scores (lower loss = more likely clean) and fit a 
2-component Beta Mixture Model to distinguish clean vs noisy samples.

This is the core of Scheme A: replace posterior-q with loss-based BMM posterior.
"""

import numpy as np
from scipy.special import digamma


class BetaMixture1D:
    """
    2-component Beta Mixture Model fitted via EM algorithm.
    
    The model assumes scores are normalized to [0, 1] where:
    - Clean samples have scores closer to 1 (low loss → high score)
    - Noisy samples have scores closer to 0 (high loss → low score)
    
    After fitting, posterior(x) returns P(clean | score=x).
    
    Based on SPR implementation with modifications for stability.
    """
    
    def __init__(self, max_iters: int = 10, tol: float = 1e-4):
        """
        Args:
            max_iters: Maximum EM iterations
            tol: Convergence tolerance for log-likelihood
        """
        self.max_iters = max_iters
        self.tol = tol
        
        # Model parameters for 2 components (clean, noisy)
        # Beta distribution params: alpha[k], beta[k]
        self.alphas = np.array([1.0, 1.0])  # shape param
        self.betas = np.array([1.0, 1.0])   # shape param
        self.weights = np.array([0.5, 0.5])  # mixing weights
        
        # Lookup table for fast posterior queries
        self.lookup = None
        self.lookup_resolution = 0.001
        
        # Track fitting status
        self.fitted = False
        self.n_samples = 0
        
    def fit(self, scores: np.ndarray, warm_start: bool = False) -> 'BetaMixture1D':
        """
        Fit the 2-component BMM to normalized scores via EM.
        
        Args:
            scores: Normalized scores in [0, 1], shape (N,)
            warm_start: If True, use current params as initialization
            
        Returns:
            self
        """
        scores = np.asarray(scores).flatten()
        # Remove exact 0s and 1s to avoid log(0)
        scores = np.clip(scores, 1e-6, 1 - 1e-6)
        self.n_samples = len(scores)
        
        if self.n_samples < 10:
            # Not enough samples, return uniform posterior
            self.fitted = False
            return self
        
        # Initialize parameters if not warm start
        if not warm_start or not self.fitted:
            self._initialize(scores)
        
        log_scores = np.log(scores)
        log_1_scores = np.log(1 - scores)
        
        prev_ll = -np.inf
        
        for iteration in range(self.max_iters):
            # E-step: compute responsibilities
            resp = self._e_step(scores)
            
            # M-step: update parameters
            self._m_step(scores, log_scores, log_1_scores, resp)
            
            # Check convergence
            ll = self._log_likelihood(scores)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        
        self.fitted = True
        # Build lookup table for fast posterior queries
        self._create_lookup()
        
        return self
    
    def _initialize(self, scores: np.ndarray):
        """Initialize parameters using score statistics."""
        # Use median split as initial guess
        median = np.median(scores)
        
        # Clean component (high scores): higher alpha, lower beta
        clean_mask = scores > median
        noisy_mask = ~clean_mask
        
        if clean_mask.sum() > 0 and noisy_mask.sum() > 0:
            clean_mean = scores[clean_mask].mean()
            noisy_mean = scores[noisy_mask].mean()
            clean_var = max(scores[clean_mask].var(), 0.01)
            noisy_var = max(scores[noisy_mask].var(), 0.01)
        else:
            clean_mean, noisy_mean = 0.7, 0.3
            clean_var = noisy_var = 0.1
        
        # Method of moments for Beta distribution
        def estimate_beta_params(mean, var):
            """Estimate alpha, beta from mean and variance."""
            mean = np.clip(mean, 0.1, 0.9)
            var = min(var, mean * (1 - mean) - 0.01)
            var = max(var, 0.001)
            common = mean * (1 - mean) / var - 1
            alpha = mean * common
            beta = (1 - mean) * common
            return max(alpha, 0.5), max(beta, 0.5)
        
        alpha_clean, beta_clean = estimate_beta_params(clean_mean, clean_var)
        alpha_noisy, beta_noisy = estimate_beta_params(noisy_mean, noisy_var)
        
        # Component 0 = noisy (low scores), Component 1 = clean (high scores)
        self.alphas = np.array([alpha_noisy, alpha_clean])
        self.betas = np.array([beta_noisy, beta_clean])
        self.weights = np.array([0.5, 0.5])
    
    def _e_step(self, scores: np.ndarray) -> np.ndarray:
        """
        E-step: compute responsibilities P(z=k | x).
        
        Returns:
            resp: (N, 2) array of responsibilities
        """
        log_resp = np.zeros((len(scores), 2))
        
        for k in range(2):
            log_resp[:, k] = (
                np.log(self.weights[k] + 1e-10)
                + self._log_beta_pdf(scores, self.alphas[k], self.betas[k])
            )
        
        # Normalize in log space for numerical stability
        log_resp_max = log_resp.max(axis=1, keepdims=True)
        log_resp = log_resp - log_resp_max
        resp = np.exp(log_resp)
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-10)
        
        return resp
    
    def _m_step(self, scores: np.ndarray, log_scores: np.ndarray, 
                log_1_scores: np.ndarray, resp: np.ndarray):
        """
        M-step: update parameters given responsibilities.
        
        Uses fixed-point iteration for Beta MLE (Newton-Raphson style).
        """
        n = len(scores)
        
        for k in range(2):
            # Effective count for component k
            n_k = resp[:, k].sum() + 1e-10
            
            # Update mixing weight
            self.weights[k] = n_k / n
            
            # Weighted sufficient statistics
            weighted_log_x = (resp[:, k] * log_scores).sum() / n_k
            weighted_log_1_x = (resp[:, k] * log_1_scores).sum() / n_k
            
            # Fixed-point updates for alpha, beta (Minka's method simplified)
            for _ in range(5):  # Inner iterations
                psi_sum = digamma(self.alphas[k] + self.betas[k])
                
                # Newton-Raphson style update
                alpha_new = self.alphas[k] * (
                    (weighted_log_x - digamma(self.alphas[k]) + psi_sum) / 
                    (psi_sum - digamma(self.alphas[k]) + 1e-10) + 1
                ) / 2
                beta_new = self.betas[k] * (
                    (weighted_log_1_x - digamma(self.betas[k]) + psi_sum) / 
                    (psi_sum - digamma(self.betas[k]) + 1e-10) + 1
                ) / 2
                
                # Clamp to valid range
                self.alphas[k] = np.clip(alpha_new, 0.1, 100)
                self.betas[k] = np.clip(beta_new, 0.1, 100)
        
        # Normalize weights
        self.weights = self.weights / (self.weights.sum() + 1e-10)
    
    def _log_beta_pdf(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Log PDF of Beta distribution."""
        from scipy.special import betaln
        return (
            (alpha - 1) * np.log(x + 1e-10)
            + (beta - 1) * np.log(1 - x + 1e-10)
            - betaln(alpha, beta)
        )
    
    def _log_likelihood(self, scores: np.ndarray) -> float:
        """Compute log-likelihood of data."""
        log_probs = np.zeros((len(scores), 2))
        for k in range(2):
            log_probs[:, k] = (
                np.log(self.weights[k] + 1e-10)
                + self._log_beta_pdf(scores, self.alphas[k], self.betas[k])
            )
        # Log-sum-exp
        max_log = log_probs.max(axis=1)
        ll = (max_log + np.log(np.exp(log_probs - max_log[:, None]).sum(axis=1))).sum()
        return ll
    
    def posterior(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute P(clean | score) for given scores.
        
        Args:
            scores: Normalized scores in [0, 1]
            
        Returns:
            Posterior probability of being clean
        """
        if not self.fitted:
            # Return score itself as fallback
            return np.asarray(scores)
        
        scores = np.asarray(scores).flatten()
        scores = np.clip(scores, 1e-6, 1 - 1e-6)
        
        # Use lookup table for speed
        if self.lookup is not None:
            indices = (scores / self.lookup_resolution).astype(int)
            indices = np.clip(indices, 0, len(self.lookup) - 1)
            return self.lookup[indices]
        
        # Direct computation
        resp = self._e_step(scores)
        # Component 1 is clean
        return resp[:, 1]
    
    def _create_lookup(self, resolution: float = 0.001):
        """Create lookup table for fast posterior queries."""
        self.lookup_resolution = resolution
        x = np.arange(resolution / 2, 1.0, resolution)
        self.lookup = self.posterior_direct(x)
    
    def posterior_direct(self, scores: np.ndarray) -> np.ndarray:
        """Direct posterior computation (without lookup)."""
        scores = np.asarray(scores).flatten()
        scores = np.clip(scores, 1e-6, 1 - 1e-6)
        resp = self._e_step(scores)
        return resp[:, 1]
    
    def predict(self, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict clean/noisy labels.
        
        Args:
            scores: Normalized scores
            threshold: Posterior threshold for clean prediction
            
        Returns:
            Boolean array (True = clean)
        """
        return self.posterior(scores) > threshold
    
    def get_clean_indices(self, scores: np.ndarray, 
                          threshold: float = 0.5,
                          min_clean_ratio: float = 0.1,
                          max_clean_ratio: float = 0.9) -> np.ndarray:
        """
        Get indices of samples predicted as clean.
        
        Includes safety bounds on clean ratio.
        
        Args:
            scores: Normalized scores
            threshold: Posterior threshold
            min_clean_ratio: Minimum fraction to keep as clean
            max_clean_ratio: Maximum fraction to keep as clean
            
        Returns:
            Array of indices
        """
        posteriors = self.posterior(scores)
        n = len(scores)
        
        # Apply threshold
        clean_mask = posteriors > threshold
        n_clean = clean_mask.sum()
        
        # Safety bounds
        if n_clean < int(min_clean_ratio * n):
            # Too few clean, take top min_clean_ratio by posterior
            k = max(1, int(min_clean_ratio * n))
            indices = np.argsort(posteriors)[-k:]
        elif n_clean > int(max_clean_ratio * n):
            # Too many clean, take top max_clean_ratio by posterior
            k = int(max_clean_ratio * n)
            indices = np.argsort(posteriors)[-k:]
        else:
            indices = np.where(clean_mask)[0]
        
        return indices


def normalize_scores(scores: np.ndarray, 
                     outlier_percentile: float = 1.0) -> np.ndarray:
    """
    Normalize scores to [0, 1] with outlier removal.
    
    Args:
        scores: Raw scores (e.g., losses)
        outlier_percentile: Remove top/bottom percentile as outliers
        
    Returns:
        Normalized scores in [0, 1]
    """
    scores = np.asarray(scores).flatten()
    
    if len(scores) == 0:
        return scores
    
    # Remove outliers
    if outlier_percentile > 0:
        low = np.percentile(scores, outlier_percentile)
        high = np.percentile(scores, 100 - outlier_percentile)
    else:
        low, high = scores.min(), scores.max()
    
    # Clamp and normalize
    scores = np.clip(scores, low, high)
    if high - low > 1e-10:
        scores = (scores - low) / (high - low)
    else:
        scores = np.ones_like(scores) * 0.5
    
    return scores


def loss_to_score(losses: np.ndarray, 
                  outlier_percentile: float = 1.0) -> np.ndarray:
    """
    Convert losses to scores (higher = more likely clean).
    
    Low loss → high score → more likely clean.
    
    Args:
        losses: Loss values
        outlier_percentile: Outlier removal percentile
        
    Returns:
        Scores in [0, 1]
    """
    # Invert: low loss = high score
    neg_losses = -np.asarray(losses)
    return normalize_scores(neg_losses, outlier_percentile)
