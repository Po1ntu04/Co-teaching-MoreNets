# Multi-Model Interactive Learning Framework (EM + SAM) – Technical Specification

## Objective

Design a multi-model interactive learning framework that extends co-teaching to **multiple models** and integrates **Sharpness-Aware Minimization (SAM)**. The goal is to robustly train on noisy labeled data by treating each sample’s cleanliness as a latent variable, and **iteratively optimizing an EM-style variational lower bound** on the data likelihood. Unlike vanilla co-teaching (two networks exchanging small-loss samples), this framework uses *M* peer models to collaboratively filter out suspected noisy samples and share gradient information for improved generalization. In summary, the design introduces a **dual-level interaction**: (1) **Loss-level interaction** – models teach each other by exchanging low-loss samples (filtering out high-loss, potentially noisy data); and (2) **Sharpness-level interaction** – models incorporate SAM by exchanging *sharpness* (gradient) information to flatten the loss landscape and escape sharp local minima. This yields an interactive learning paradigm that converges toward flatter minima (better generalization) and outperforms standard co-teaching in the presence of noisy labels.

## Variables and Notation

The following table summarizes key variables and parameters used in the specification, including their meaning, dimensions, and value ranges:

| **Variable**                            | **Shape / Range**                    | **Description**                                              |
| --------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **D**                                   | $N$ samples = ${(x_i, y_i)}_{i=1}^N$ | Training dataset of size $N$. Each sample $x_i$ (input features) with observed label $y_i$. Some labels may be noisy (incorrect). |
| **M**                                   | Integer ($\ge 2$)                    | Number of models (neural networks) in the interactive ensemble. For example, $M=3$ for three peer networks. All models have identical architecture (for simplicity) unless otherwise specified. |
| **$\theta_m$**                          | Depends on model architecture        | Parameters (weights) of model $m$, for $m=1,\dots,M$. Each $\theta_m$ is updated during training. Initially $\theta_m^{(0)}$ are random or pre-trained. |
| **$Z_i$**                               | ${0,1}$ (latent binary variable)     | Latent indicator for sample $i$ being **clean** ($Z_i=1$) or **noisy** ($Z_i=0$). A “clean” sample means $y_i$ is correct; “noisy” means the label is corrupted. These are unobserved ground-truth indicators for each sample. In EM terms, $Z_i$ are the hidden variables. |
| **$q_i(t)$** or **$q(Z_i=1)$**          | $[0,1]$ (probability)                | The **variational probability** (responsibility) that sample $i$ is clean at iteration $t$. Denoted $q_i(t)=q(Z_i=1)$, with $q_i(t)+ (1-q_i(t))=1$. These are computed in the E-step (soft version) or derived from a hard selection. Collectively $q(Z)$ denotes the factorized distribution $\prod_i q_i$ over all latent $Z$. |
| **$S_m(t)$**                            | Subset of ${1,\dots,N}$              | The **selected clean subset** indices chosen by model $m$ at iteration $t$. $S_m(t)$ typically contains a fraction $R(t)$ of the dataset (those with lowest loss according to model $m$). For a mini-batch version, $S_m(t)$ refers to selected indices within that batch. If using soft $q$, $S_m(t)$ may represent the top fraction (hard selection) chosen by model $m$. |
| **$R(t)$**                              | $0 < R(t) \le 1$                     | **Retain rate** at iteration $t$ – the fraction of data to keep as “clean”. This schedule is predetermined or annealed over time. For example, start $R(0)\approx 1$ and gradually decrease to the expected clean data fraction (e.g. $1 - \text{noise rate}$). *E.g.* $R(t) = 1 - \min{t/T_k,\ \tau}$ as used in co-teaching, where $\tau$ is the target final fraction of clean data. |
| **$\mathcal{L}_m(i)$** or **$L_{m,i}$** | Real (e.g. $\mathbb{R}_{\ge0}$)      | Loss of model $m$ on sample $i$. For instance, $\mathcal{L}*m(i) = \ell(f*{\theta_m}(x_i), y_i)$, where $\ell$ is the per-sample loss (e.g. cross-entropy). High $\mathcal{L}_m(i)$ suggests sample $i$ is likely mislabeled relative to model $m$’s current knowledge. |
| **$\lambda_m(t)$**                      | $[0,1]$ (real) or ${0,1}$            | **Reliability weight** for model $m$ at iteration $t$. $\lambda_m(t)=1$ means model $m$ is fully trusted/active; $\lambda_m(t)=0$ means the model is pruned (removed). This parameter is used in model selection/aggregation – e.g. a lower $\lambda_m$ reduces model $m$’s influence on consensus. $\lambda_m(t)$ is updated based on model performance (could be gradually decayed or set to 0 for pruning). |
| **$\rho$**                              | Small positive real (e.g. $0.05$)    | SAM perturbation coefficient (step size for sharpness). Controls the radius of weight perturbation for SAM’s inner maximization step. Sometimes noted as $\rho$ or $\alpha$ in SAM literature. |
| **$\hat{\epsilon}(\theta_m)$**          | Same shape as $\theta_m$             | **SAM perturbation vector** for model $m$’s weights. Computed as $\hat{\epsilon}(\theta_m) = \rho \cdot \frac{\nabla_{\theta_m} \mathcal{L}*{S*{\neg m}}(m)}{|\nabla_{\theta_m} \mathcal{L}*{S*{\neg m}}(m)|*2}$ (gradient direction on a certain data subset, scaled to norm $\rho$). This is added to $\theta_m$ during SAM’s inner step. ($S*{\neg m}$ denotes data selected by other models for $m$, see below.) |
| **$\text{ELBO}(t)$**                    | Real                                 | Evidence Lower Bound at iteration $t$ – a proxy for the objective being maximized. In this context, $\text{ELBO}(t) = L({\theta_m}, q(t))$ is the variational lower bound on $\log p(D |

**Notes:** ${\theta_m}*{m=1}^M$ may be collectively denoted $\Theta$. $S*{\neg m}(t)$ will be used to denote the aggregated selection (or distribution $q$) that model $m$ uses for updating at iteration $t$, typically derived from **other** models (excluding $m$ itself). Finally, $\nabla_{\theta}\mathcal{L}(f_\theta, D')$ denotes the gradient of loss on dataset $D'$ w.rt. parameters $\theta$. Each model’s architecture and loss $\ell$ (e.g. softmax cross-entropy for classification) are assumed given.

## E-step: Inferring $q(Z)$ (Clean Sample Probabilities)

**Goal:** Compute the posterior probability or indicator that each sample is clean (latent $Z_i=1$) given the current models. In EM terms, we want $q_i(t) = \Pr(Z_i=1 \mid D, \Theta^{(t)})$ – the responsibility that sample $i$ is noise-free at iteration $t$. We describe two approaches:

- **Soft E-step (Probabilistic):** We assign a **soft probability** $q_i(t) \in [0,1]$ for each sample being clean. Intuitively, a sample should have a higher $q_i$ if most models agree it has a low loss (fits well), and a lower $q_i$ if it yields high loss or inconsistent predictions across the ensemble. One way to derive $q_i(t)$ is via Bayes’ rule on a simple noise model: for example, assume each model’s output provides evidence for $Z_i$. A practical implementation: use the **aggregated losses or confidences** of all models on sample $i$. Let $L_{m,i}^{(t)} = \mathcal{L}_m(i)$ be model $m$’s loss on $(x_i,y_i)$. Define an **aggregate loss score** $A_i(t)$ (smaller means more likely clean) using one of the strategies below (see **Aggregation**). Then convert that to a probability. For instance:

  qi(t)=σ(−Ai(t)−Tpivotγ)q_i(t) = \sigma\Big( -\frac{A_i(t) - T_{\text{pivot}}}{\gamma} \Big)qi(t)=σ(−γAi(t)−Tpivot)

  where $\sigma$ is the sigmoid function, $T_{\text{pivot}}$ is a threshold (e.g. median aggregated loss) and $\gamma$ a scaling factor. This yields $q_i(t)\approx1$ for samples with much-below-threshold loss and $q_i(t)\approx0$ for high-loss samples, with a smooth transition. Alternatively, one can use a softmax over all samples’ negative losses to interpret $q_i$ as a normalized weight:

  qi(t)=exp⁡(−Ai(t)/T)∑j=1Nexp⁡(−Aj(t)/T),q_i(t) = \frac{\exp(-A_i(t)/T)}{\sum_{j=1}^N \exp(-A_j(t)/T)},qi(t)=∑j=1Nexp(−Aj(t)/T)exp(−Ai(t)/T),

  where $T$ is a temperature hyperparameter. This makes $q_i(t)$ large if sample $i$’s aggregated loss is relatively the lowest among the dataset. In essence, the soft E-step treats $q_i(t)$ as the *posterior probability of sample $i$ being clean*, proportional to how well current models predict $y_i$ vs. treating $y_i$ as random noise. If models output class-probabilities, another approach is to use their predicted likelihood of the given label: e.g. $q_i(t) = \frac{1}{M}\sum_{m=1}^M p_{\theta_m}^{(t)}(y_i|x_i)$ (average confidence that $y_i$ is correct). This average or product of confidences can serve as $\Pr(Z_i=1|\Theta)$ estimate (product would assume model independence). **Normalization:** Ensure $q_i(t)$ is clipped to $[0,1]$ (or re-normalized across all $i$ if using a softmax scheme).

- **Hard E-step (Thresholding/Selection):** Instead of a fractional $q_i$, we perform hard assignment by selecting a subset of samples deemed clean. This corresponds to setting $q_i(t) = 1$ for selected samples and $0$ for others, based on a threshold on losses. The typical strategy (from co-teaching) is: each model *independently* picks the fraction $R(t)$ of samples with smallest loss in the current batch. In a multi-model setting, we can refine this by requiring **consensus or multi-model criteria** (see **Aggregation** below). For example, define $A_i(t)$ as an aggregate loss or score for sample $i$. Then choose the set $S(t) = \text{argsmallest}*{i}(R(t) \cdot N)$ based on $A_i(t)$ – i.e. select the lowest $R(t)$ fraction of samples by aggregated loss. All selected $i \in S(t)$ get $q_i(t)=1$ (clean), others get $0$. Alternatively, use a fixed loss threshold: e.g. $q_i(t)=1$ if $A_i(t) < \delta(t)$ for some cutoff $\delta(t)$, ensuring approximately the desired fraction. Each model $m$ could also have its own selection $S_m(t)$ (e.g. each picks top $R(t)$ based on its perspective $L*{m,i}$), and then we later combine these selections (union, intersection, etc.) as described next.

**Output of E-step:** an updated **clean probability vector** $q(t) = (q_1(t),\dots,q_N(t))$ or equivalently a set $S(t)$ of samples considered clean at iteration $t$. In EM terms, this $q(t)$ maximizes the ELBO $L(\Theta^{(t)}, q)$ given fixed model parameters, by setting $q(Z) = p(Z|D,\Theta^{(t)})$. In practice, **when implementing hard selection, $q_i(t)$ can be taken as binary** (this is a **hard E-step** approximation). When using soft $q$, we will incorporate $q_i(t)$ as weights in the M-step loss. Both approaches aim to reflect each sample’s likelihood of being a clean example under current models.

## Aggregation of Multi-Model Outputs (Consensus Strategy)

With multiple models producing loss or confidence estimates, we need to **aggregate their outputs** to decide which samples are clean. A robust aggregation reduces bias from any single model and uses the ensemble’s collective wisdom. We propose several methods:

- **Mean Loss (Average):** For each sample $i$, compute the average loss across models:
   Ai(t)=1M∑m=1MLm,i(t).A_i(t) = \frac{1}{M}\sum_{m=1}^M L_{m,i}^{(t)}.Ai​(t)=M1​∑m=1M​Lm,i(t)​.
   Then use $A_i(t)$ as the aggregate score (lower is better). This treats all models equally and works under the assumption that clean samples yield low loss on most models.
- **Median Loss:** Compute $A_i(t)$ as the median of ${L_{m,i}^{(t)} : m=1..M}$. Median is more robust to outlier models – if one model’s loss is an outlier (perhaps the model is temporarily wrong about $i$), it won’t overly influence the score. This is useful if occasional model mistakes should not drop a genuinely clean sample.
- **Trimmed Mean:** Similar to median, but you can remove the highest and lowest $p%$ of losses and average the rest. For example, drop the top and bottom 1 model’s losses and average the remaining $M-2$. This can balance between mean and median, ignoring extreme outliers.
- **Majority Vote (for Hard Selection):** Each model $m$ can mark a sample $i$ as “clean” if $L_{m,i}$ is among its lowest $R(t)$ fraction. Then let
   vi=#{m:i∈Sm(t)}v_i = \#\{m : i \in S_m(t)\}vi​=#{m:i∈Sm​(t)}
   be the number of models that selected sample $i$. We can set a threshold $h$ (e.g. $h = \lceil M/2 \rceil$ for majority) and declare $i$ clean if $v_i \ge h$. This yields a consensus set $S_{\text{consensus}}(t) = { i : v_i \ge h }$. The equivalent soft $q_i$ could be $v_i/M$ (the fraction of models voting clean).
- **Reliability-Weighted Aggregation:** If certain models are more trustworthy (tracked via $\lambda_m(t)$), weight their contributions more. For instance,
   Ai(t)=1∑mλm(t)∑m=1Mλm(t) Lm,i(t).A_i(t) = \frac{1}{\sum_{m}\lambda_m(t)}\sum_{m=1}^M \lambda_m(t)\, L_{m,i}^{(t)}.Ai​(t)=∑m​λm​(t)1​∑m=1M​λm​(t)Lm,i(t)​.
   A model with lower reliability $\lambda_m$ contributes less to the score. In a vote scheme, you might require a weighted sum of votes $\sum_m \lambda_m I[i\in S_m(t)]$ to exceed a threshold. The weights $\lambda_m(t)$ evolve as models prove themselves (see **Absorption/Pruning**).
- **Label Prediction Agreement:** Instead of losses, consider each model’s predicted label $\hat{y}_m(x_i)$. Clean samples likely yield *consensus in predictions*. We could aggregate by checking how many models predict the given label $y_i$. Let $u_i = #{ m : \hat{y}_m(x_i) = y_i }$. If $u_i$ is high (most models agree on the provided label), consider $i$ clean. This is akin to a vote on label correctness. Conversely, if models disagree or mostly predict a class different from $y_i$, it’s a sign $y_i$ may be wrong (noisy). This method can supplement loss-based selection.

**Selecting/Weighting with $A_i(t)$:** Once an aggregation strategy produces $A_i(t)$ or an equivalent score for each sample, we either **(a)** choose the lowest $N \cdot R(t)$ samples (for hard E-step) or **(b)** compute soft weights $q_i(t)$ from the scores as described. Pseudocode for hard selection with average loss might look like:

```
# Given current losses L_{m,i} for all models m and samples i in batch:
A = [mean(L[:, i]) for i in batch_indices]  # aggregate loss
selected_indices = argsort(A)[:ceil(R(t) * batch_size)]
for i in batch_indices:
    q[i] = 1 if i in selected_indices else 0
```

For soft selection, one might do:

```
A = [mean(L[:, i]) for i in batch_indices]
# Map aggregate losses to probabilities (example using negative exponential):
q_probs = softmax(-np.array(A) / T)
```

This yields $q_i(t)$ values in (0,1).

**Note:** The design allows flexibility in aggregation. For example, early in training when models are still learning, a simple mean might suffice; later, as $\lambda_m$ values diverge or some models become erratic, a weighted or median approach could be safer. **By default, we recommend using the mean or median of losses** for simplicity, and incorporate reliability weights $\lambda_m$ if model performance diverges significantly.

## M-step: Parameter Update with $q$-Weighted Loss

After estimating $q(Z)$ (the distribution of clean vs noisy for each sample), we update each model’s parameters $\theta_m$ to maximize the expected log-likelihood of the data under those assignments (i.e. maximize the ELBO). In practice, this means **training each model on the samples deemed clean (with appropriate weights)**. Key points for the M-step:

- **Using $q$ as sample weights:** For each model $m$, we define a $q$-weighted loss over the training set (or mini-batch) and take a gradient step to minimize it. For example, if using soft probabilities, the loss for model $m$ at iteration $t$ is:
   Lm(t)(θm)=1N∑i=1Nqi(t)  ℓ(fθm(xi),yi).\mathcal{L}_m^{(t)}(\theta_m) = \frac{1}{N}\sum_{i=1}^N q_i(t)\;\ell(f_{\theta_m}(x_i), y_i).Lm(t)​(θm​)=N1​∑i=1N​qi​(t)ℓ(fθm​​(xi​),yi​).
   Here $q_i(t)$ weights the contribution of sample $i$. If $q_i(t)=0$, sample $i$ (likely noisy) is effectively ignored; if $q_i(t)\approx1$, $i$ is treated as trustworthy. Model $m$ then does a gradient update: $\theta_m \gets \theta_m - \eta \nabla_{\theta_m} \mathcal{L}*m^{(t)}$ (for learning rate $\eta$). This corresponds to maximizing $E*{q(t)}[\log p(D,Z|\theta_m)]$ as in EM.

- **Cross-model teaching (for hard selection):** In the multi-model interactive setting, to avoid confirmation bias, each model should learn from others’ selections rather than reinforcing its own bias. Implement this by **cross-utilizing the clean sets**: Model $m$ updates using data selected *by the other models*. For instance, if using majority vote selection $S_{\text{consensus}}(t)$, all models could use that common set. Alternatively, in a leave-one-out style: let $S_{\neg m}(t)$ be the subset of samples that at least $M-1$ (all except model $m$) agree on as clean, or an aggregate excluding $m$’s own judgment. Then model $m$ uses $S_{\neg m}(t)$ for its update. In practice with hard selection, this means model $m$’s mini-batch gradient is computed on mini-batch $\cap S_{\neg m}(t)$ (samples picked by others). For soft $q$, a similar effect is achieved if the aggregator omitted model $m$ (e.g. average of others’ losses) when computing $q_i$ for use in model $m$’s loss. This way, model $m$ does not simply see its own evaluation of which samples are clean, but relies on peers, reducing self-reinforcement of errors.

- **Gradient update formula (two-model example):** For context, in co-teaching with 2 models $f$ and $g$, the M-step updates were: $\theta_f \leftarrow \theta_f - \eta \nabla_{\theta_f} \sum_{i \in S_g} \ell(f(x_i),y_i)$ and similarly $\theta_g$ on $S_f$. This is equivalent to $\theta_f^{(t+1)} = \arg\max_{\theta_f} E_{q^{(t)}*g}[\log p(Z^g, D|\theta_f,\theta_g^{(t)})]$. We generalize this to $M$ models by using $S*{\neg m}(t)$ or aggregated $q(t)$ excluding $m$ for model $m$’s update.

- **Mini-batch implementation:** In practice, we operate on mini-batches. For each mini-batch $B \subset D$, we compute $q_i(t)$ or select samples in $B$, then update each model on that batch’s clean samples. Pseudocode (simplified) per iteration:

  ```
  for each model m=1..M:
      # Determine which samples in batch B to use for model m:
      if hard_selection:
          B_clean_for_m = { i in B : i ∈ S_{≠m}(t) }   # selected by others
      else:
          # soft: use weights, possibly excluding m's influence
          weights = [q_i(t) from aggregator excluding m for i in B]
          B_clean_for_m = B with sample weights = weights
      # Compute gradient of loss on B_clean_for_m:
      grad = ∇_{\theta_m} (1/|B|) * sum_{i in B} [w_i * ℓ(f_{\theta_m}(x_i), y_i)]
      θ_m = θ_m - η * grad
  ```

  If using hard $S_{\neg m}$, $w_i$ is effectively 1 for $i \in S_{\neg m}(t)$ and 0 otherwise. If using soft, $w_i = q_i(t)$ (with or without $m$’s own contribution as decided). Each model is updated in parallel or sequentially (order doesn’t matter if using synchronized $q(t)$ from the E-step).

- **Ensuring convergence:** Each M-step should *approximately* maximize the lower bound $L(\Theta,q)$ for fixed $q$. In practice, one or a few SGD steps are taken rather than full maximization, since we intermix E and M steps per mini-batch. The process iterates until convergence criteria (e.g. epoch count or ELBO stabilization). This EM-like alternating update has been theoretically justified to converge to at least a local optimum【12†L149-158】.

## Absorption and Pruning Mechanism (Model Adaptation)

As training progresses with multiple models, we expect some models to perform better or converge to similar functions. Maintaining redundant or poor models is inefficient. We introduce an **absorption/pruning mechanism** governed by $\lambda_m(t)$ to adapt the ensemble size over time:

- **$\lambda_m(t)$ as reliability score:** Each model $m$ has a weight $\lambda_m(t) \in [0,1]$ indicating its reliability or contribution. Start with $\lambda_m(0)=1$ for all (all models fully active). Over time, if model $m$ consistently underperforms or becomes redundant, we will decrease $\lambda_m$. This can be done gradually (continuous) or in a binary fashion (set to 0 = prune). Several criteria can inform $\lambda$ updates:
  - **Validation performance:** If a small clean validation set is available, track each model’s accuracy. Any model falling significantly behind the best could be downgraded (lower $\lambda$).
  - **Agreement with peers:** Compute the correlation or agreement of model $m$ with the ensemble’s consensus. For example, how often does model $m$’s selected set $S_m(t)$ differ from the consensus $S_{\text{consensus}}(t)$? Or how often does it predict the same labels as others for the training data? If a model disagrees frequently (potentially due to it not learning or diverging), that model might be unreliable. Conversely, if a model simply mirrors another (high agreement), it might be redundant.
  - **Training loss / gradient magnitude:** If one model’s training loss remains high relative to others or its gradients are unstable, it may be stuck in a bad local minima. This could trigger a reduction in trust.
- **Update rule for $\lambda_m$:** We define a schedule to adjust $\lambda_m$. Example approach: every epoch, evaluate model performance metrics (one or combination of above). Then:
  - If model $m$ is flagged as *poor*, set $\lambda_m(t+\Delta t) = \lambda_m(t) \times \beta$ for some decay factor $0<\beta<1$ (e.g. $\beta=0.5$) to reduce its influence. If $\lambda_m$ falls below a small threshold $\epsilon_{\min}$, **prune** model $m$ (remove it from ensemble and stop updating it).
  - If model $m$ performs on par with the best, keep $\lambda_m \approx 1$. We might even set $\lambda_m(t)=1$ for all actively retained models for simplicity, only dropping or keeping models (binary prune/retain).
  - Optionally, if two models become very similar (e.g. their outputs differ on <1% of samples), one could be absorbed: remove one model and optionally average its weights into the remaining model (knowledge transfer). However, a simpler approach is just to drop it, as other models have essentially learned the same function.
- **Absorption step timing:** Check for pruning at certain milestones (e.g. every few epochs or at the end of training). We don’t want to prune too early (could lose complementary knowledge), so one heuristic: wait until the noise selection has mostly stabilized (i.e. after $R(t)$ has reached the final value and models have trained on that for a while). Then prune any model that lags significantly or is redundant. This can reduce computation and potentially improve consensus (less conflicting signals). Ensure at least one model remains (if all but one are pruned, we essentially return to a single model scenario).
- **Using $\lambda_m$ during training:** While models are still active, we can use $\lambda_m(t)$ in aggregation (as discussed) to weight each model’s vote. For instance, if $\lambda_3$ has decayed to 0.5 while others are 1.0, model 3’s losses count half as much in the mean, or its vote counts half. This smooths the transition – unreliable models gradually influence the selection less, even before full removal.
- **Logging $\lambda_m$:** It’s important to log or visualize $\lambda_m(t)$ for each model (see **Logging**). A sharply dropping $\lambda_m$ indicates a model being phased out. Ideally, the remaining models should pick up any slack (which usually they do if they have similar knowledge by that time).

In summary, this mechanism ensures the framework is **dynamic**: it can shrink to the most competent models by the end of training, absorbing knowledge into a smaller, strong ensemble. In code, one might implement a function `update_reliabilities(models, performance_metrics)` to adjust the $\lambda$ array each epoch.

## SAM Integration Hook (Sharpness-Aware Updates)

To integrate **Sharpness-Aware Minimization (SAM)** into the training, we modify the weight update steps to include a perturbation that seeks a flatter minimum. The SAM procedure introduces an inner maximization: before the usual gradient descent, find an adversarial perturbation in weight space that maximizes the loss (within a small norm). Then update the weights to minimize the loss at that perturbed point. Our framework will apply SAM **at each M-step update for each model** (second level of interaction):

1. **Compute peer-selected data:** Determine the data that model $m$ will train on this step (e.g. $D_{\text{train},m} = {(x_i,y_i) : i \in S_{\neg m}(t)}$ if using hard selection, or include all with weights $q_i$). Let’s denote this $D_m$ for brevity.
2. **Gradient for perturbation (inner step):** Compute $\nabla_{\theta_m} \mathcal{L}(f_{\theta_m}, D_m)$, the gradient of model $m$’s loss on its training subset. This gradient indicates the direction in weight space that *increases* the loss on $D_m$. Normalize this gradient and scale by $\rho$ (the SAM radius factor) to obtain $\hat{\epsilon}(\theta_m)$:
    ϵ^(θm)=ρ∇θmL(fθm,Dm)∥∇θmL(fθm,Dm)∥2.\hat{\epsilon}(\theta_m) = \rho \frac{\nabla_{\theta_m} \mathcal{L}(f_{\theta_m}, D_m)}{\|\nabla_{\theta_m} \mathcal{L}(f_{\theta_m}, D_m)\|_2}.ϵ^(θm​)=ρ∥∇θm​​L(fθm​​,Dm​)∥2​∇θm​​L(fθm​​,Dm​)​.
    This corresponds to Algorithm 1 lines 9-11: computing $\hat{\epsilon}(\theta_f)$ and $\hat{\epsilon}(\theta_g)$ using the gradient on peer-selected data. In code, this involves performing a forward pass on $D_m$, getting the loss, backward to get grads, then doing vector normalization.
3. **Perturb weights:** Temporarily update model $m$’s weights by adding this perturbation: $\theta_m^{\text{temp}} = \theta_m + \hat{\epsilon}(\theta_m)$. This is *not* a permanent update, but a step to evaluate the worst-case loss increase.
4. **Compute sharpness-aware gradient (outer step):** Evaluate the loss of model $m$ on $D_m$ at the perturbed weights $\theta_m^{\text{temp}}$. Compute the gradient $\nabla_{\theta} \mathcal{L}(f_{\theta}, D_m)\big|*{\theta=\theta_m^{\text{temp}}}$. This gradient (call it $G_m$) is an “approximate worst-case” gradient that accounts for how the loss would increase in a local neighborhood of the current weights. For example, $G_f = \nabla*{\theta_f} \mathcal{L}(f_{\theta}, D_{\neg f})\big|_{\theta_f + \hat{\epsilon}(\theta_f)}$ in Algorithm 1.
5. **Restore weights and update:** Remove the temporary perturbation (set $\theta_m$ back to original pre-perturbation values, if they were overwritten, or use a copy). Then perform the actual model update using the computed sharpness-aware gradient: $\theta_m \leftarrow \theta_m - \eta , G_m$. This update will push $\theta_m$ to minimize the loss in the neighborhood of the sharpest direction, thereby favoring flatter minima. In practice, one can combine steps 3-5 by directly computing $G_m$ without permanently modifying $\theta_m$ (just use the perturbed values in the gradient computation, then discard them when applying the update).

**Implementation alternatives:**

- Use an existing SAM optimizer implementation: Many deep learning libraries provide SAM as a wrapper optimizer (which automates the two-step process). One can simply plug in a SAM optimizer for each model with the desired $\rho$. Ensure that the SAM’s loss is computed on the appropriate dataset ($D_m$ for each model).
- Manual implementation: As outlined, manually compute gradients twice per batch. This roughly doubles the computational cost of training (each mini-batch does two forward-backward passes per model). However, since $R(t)$ often drops and we might use smaller subsets, the overhead is manageable.
- **Frequency of SAM:** In some scenarios, one might not apply SAM at every single update (to save compute). For example, apply SAM every $k$ iterations, or only after a certain epoch. But the default is to apply it consistently to fully reap generalization benefits.
- **Hyperparameters:** $\rho$ controls the perturbation radius – small $\rho$ might yield negligible effect; too large $\rho$ could over-regularize. Typical $\rho$ values are in $[0.05, 0.5]$ depending on dataset and model. Additionally, one might tie $\rho$ to $R(t)$ or iteration count (e.g. a larger $\rho$ once noise filtering is strict, to aggressively flatten). This is an advanced option; generally use a fixed $\rho$.

By incorporating SAM in this way, the training becomes a **dual-level optimization**: first level filtering out high-loss samples, second level exchanging “sharpness” information to flatten the landscape. This combined approach, termed Sharpness-Reduction Interactive Teaching (SRIT), has been shown to achieve better generalization and resist getting stuck in poor local optima. Figure 1 in the paper illustrates how the loss landscape becomes gradually flatter through the iterations with SAM.

## Logging and Monitoring Requirements

To ensure the algorithm is functioning correctly and to analyze its behavior, we will log several metrics during training:

- **ELBO / Loss Lower Bound:** Track an ELBO proxy $L(\Theta(t), q(t))$ over iterations. Even if computing the exact ELBO is difficult (due to unknown true noise distribution), we can log the *average weighted loss* as a stand-in. For instance: $-\frac{1}{N}\sum_i q_i(t)\log p(y_i|x_i,\Theta(t))$ could serve as a proxy (lower is better). If using hard selection, track the average loss on selected vs. non-selected sets. This indicates if the models are indeed focusing on an easier (cleaner) portion of data and if that yields a lower overall loss bound. We expect the ELBO proxy to **increase (or at least not decrease)** over EM iterations; monitoring it helps detect divergence.
- **$q(Z)$ Evolution:** Monitor statistics of the clean probabilities $q_i(t)$. For example, log the **average $q_i$**, or the **histogram of $q_i$ values** each epoch. In hard selection mode, this is essentially the fraction of data selected ($R(t)$ by design) and which samples are selected. We can track how many times each sample was selected across epochs (e.g. some sample gets consistently low $q$, likely truly noisy). For soft $q$, we can track the distribution – ideally it will become bimodal (samples split into confidently clean $q\approx1$ and confidently noisy $q\approx0$ groups) as the models converge. Any oscillation in $q_i$ (samples switching frequently) could indicate instability that might need adjusting hyperparameters.
- **Model Agreement/Correlation:** Because interactive learning relies on model diversity (especially early on), we log how similar the models are. Metrics include:
  - **Prediction agreement:** fraction of samples on which all models predict the same label (on training data or a validation set). Also pairwise agreement percentages.
  - **Selection overlap:** for each pair of models $(m, m')$, compute $|S_m(t) \cap S_{m'}(t)| / |S_m(t)|$ – the fraction of model $m$’s selected samples that model $m'$ also selected. High overlap approaching 1 too early might indicate the models have become too similar (less effective co-teaching). Ideally, early on this overlap is moderate (diversity), and later it increases as they converge on the true clean set.
  - **Ensemble diversity:** measures like entropy of the ensemble predictions or variance in losses among models on the same sample. We can log the mean of $\text{std_dev}{L_{m,i}}_{m}$ across i, to see if the models’ loss estimates are converging.
- **Model Reliability $\lambda_m$ and Pruning Events:** Plot $\lambda_m(t)$ for each model over time. This will show if any model is being phased out. If we implement pruning, log when it happens (iteration and which model). Also log any merging/absorption details if applicable (e.g. “Model 3 pruned at epoch 20 (low accuracy)” or “Model 2 and 4 found redundant, removed 4 at epoch 15”). This helps in analyzing if the right decisions were made. If a model is pruned, continue logging its $\lambda$ as 0 or mark it as inactive.
- **Training/Validation Accuracy:** Even though training data is noisy, we should monitor how each model and the ensemble (e.g. majority vote of models) performs on a hold-out *clean* validation set or on the noisy training set. Validation accuracy on clean data is the ultimate metric to see if generalization is improving. The ensemble prediction accuracy (e.g. majority vote of M models) can also be tracked on the training set to see how it fits noisy labels (hopefully it does *not* fit the noise, maintaining lower training accuracy than a naive model would, which is a good sign of noise filtering).
- **Gradients / Sharpness metrics:** Optionally monitor the norm of gradients before and after SAM perturbation for each model, or the magnitude of $\hat{\epsilon}(\theta_m)$. For instance, log $|\nabla L_{D_m}(f_{\theta_m})|$ vs $|G_m|$ to confirm SAM is altering updates as expected (usually $G_m$ might be smaller in norm but directs to flatter region). We could also monitor the sharpness of the solution by checking difference in loss with and without small perturbations (an indirect measure of flatness).

All logs should be timestamped by epoch or iteration. For easier analysis, one can create plots: e.g. *ELBO vs epoch*, *average q vs epoch*, *accuracy vs epoch*, *each $\lambda_m$ vs epoch*, etc. These will be invaluable for debugging (e.g. if $q$ collapse or oscillates, if one model dominates, if SAM is causing any slowdowns, etc.).

In implementation, a logging dictionary or CSV can be maintained. For example, at end of each epoch, record:

```
{
  "epoch": t,
  "ELBO": current_elbo,
  "mean_q": avg_q,
  "std_q": std_q,
  "model_agreement": agreement_rate,
  "model1_lambda": lambda_1,
  "model2_lambda": lambda_2,
  ...
  "model1_val_acc": val_acc_1,
  "ensemble_val_acc": val_acc_ens,
  ...
}
```

This structured logging will allow easy analysis after training.

## Demo Setup (CIFAR-10 Noisy-Labels Prototype)

To validate the complete pipeline, we outline a prototype experiment on a small-scale vision task:

**Dataset & Noise:** Use the CIFAR-10 dataset (50,000 train, 10,000 test images, 10 classes). Introduce **40% noise** in the training labels. For example, for *symmetric noise*, randomly flip 40% of labels to a uniform random incorrect class. (Alternatively, *class-conditional noise* where certain classes might be confused with specific others could be tried; but symmetric is straightforward.) The test set remains clean to evaluate generalization.

**Model Architecture:** For speed, use a moderate CNN, e.g. a 9-layer convolutional network or ResNet-18. Use the same architecture for all $M$ models. For demonstration, let’s choose **M = 3** networks. (M=2 would reduce to standard co-teaching; M=3 shows multi-model benefit.)

**Training Hyperparameters:** For instance, train for 100 epochs with SGD or Adam optimizer. Learning rate $\eta$ might start at 0.1 with decay. Use a co-teaching schedule for $R(t)$: e.g. $R(0)=1.0$ and linearly decay to $R=0.6$ by epoch 20 (since we expect 60% clean data with 40% noise), then keep $R(t)=0.6$ for the remaining epochs. This schedule follows the original co-teaching practice. Batch size can be 128.

**SAM Parameters:** Set $\rho$ (SAM perturbation factor) to e.g. 0.1. This means each update will make a 0.1-sized step in weight space towards increasing loss. This value can be tuned; smaller if the model is large or noise is high (to not overshadow the selection mechanism). Use SAM at every update (the overhead for 3 models is roughly 6 forward-backward passes per batch, which is acceptable on modern GPUs). If computation is a concern, one can apply SAM every other batch or every epoch as a simplification.

**Procedure:** Initialize 3 models with different random seeds. At each training iteration:

1. Fetch next mini-batch $B$.
2. Each model computes losses $L_{m,i}$ for $i \in B$.
3. **E-step:** Compute an aggregated clean probability or select clean subset from $B$. For simplicity in code, implement the **hard selection** variant: each model picks its $R(t)$ fraction of lowest-loss samples in $B$ (small-loss trick), then take the intersection or majority vote among the 3 sets to decide final clean batch $B_{\text{clean}}$ (this is $S_{\text{consensus}}(t)$ for the batch). Alternatively, use the average loss method: compute mean loss per sample and pick the top $R(t)$ fraction in $B$. (We can experiment with both and see which is more stable.)
4. **M-step:** For each model $m$:
   - Take $B_{\text{clean}}$ (or the subset selected excluding $m$’s own selection if doing leave-one-out consensus).
   - Perform the SAM update: compute gradient on $B_{\text{clean}}$, perturb weights, compute gradient at perturbed weights, update weights.
5. Update iteration count. If end of epoch, update $R(t)$ according to schedule.
6. Every epoch, evaluate on a *clean validation set* (could set aside 5k clean CIFAR-10 images as validation or use test for proxy) to get each model’s accuracy and ensemble accuracy. Log the metrics.

**Expectations:** We anticipate that in the first few epochs, all models will train on nearly all data ($R \approx 1$), possibly fitting some noise. As $R(t)$ decreases and the models start selecting data, they will drop the high-loss (likely noisy) samples. Each model might drop different samples initially, maintaining some diversity. By epoch ~20, $R(t)$ reaches 0.6, so about 60% of the data is retained per batch – ideally the truly clean portion if the method works. With SAM in play, the models should converge to parameters that lie in flatter regions of the loss surface, improving their generalization. We expect the ensemble to significantly outperform a single model trained normally on noisy data. Specifically, prior work shows SRIT (our method) yields higher accuracy than co-teaching alone. For example, on CIFAR-10 with 40% noise, co-teaching might reach ~77% accuracy, whereas with SAM (SRIT) one could see ~80% or more. Our 3-model setup could further improve robustness slightly via the ensemble effect.

**Validation:** After training, evaluate on the clean CIFAR-10 test set: measure each model’s accuracy and the majority-vote ensemble accuracy. Also, verify the noise filtering effect: check some of the training samples that were consistently given low $q_i$ (dropped early) – those should correspond to the injected noisy labels. We can calculate precision/recall of noise identification if we know which labels were corrupted. A successful run will identify a large fraction of noisy samples while retaining most clean ones.

**Example outcome:** In a trial, suppose our method achieves ~78% test accuracy, compared to ~75% with standard 2-model co-teaching and ~70% with standard training on 40% noisy data. The logging might show the ELBO proxy increasing after an initial dip, $q_i$ distribution separating (many $q_i$→1, many →0), and perhaps one model’s $\lambda$ was reduced if it lagged. We would include a brief analysis of these logs to conclude the demo, confirming that the framework indeed **activated interactive teachings with sharpness-awareness** as intended.