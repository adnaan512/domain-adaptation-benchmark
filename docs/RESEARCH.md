# Research Methodology

## Domain Shift and Test-Time Adaptation

### Problem Statement

Models trained on clean data (source domain *S*) routinely fail when deployed
on data drawn from a different distribution (target domain *T*) — a phenomenon
called **domain shift** or **distribution shift**.  In computer vision, domain
shift arises from:

- **Image corruptions**: noise, blur, compression artefacts, weather effects
- **Sensor differences**: camera models, lighting conditions, focus settings
- **Operational changes**: new geographic locations, seasonal variation

This is not a corner case.  Autonomous vehicles trained in clear weather
encounter rain, fog, and glare during deployment.  Medical imaging models
trained at one hospital encounter different scanner calibrations at another.

### Why Test-Time Adaptation?

Conventional solutions require labelled data from the target domain (supervised
domain adaptation) or access to both domains during training (domain
generalisation).  **Test-Time Adaptation (TTA)** relaxes both requirements:

- ✓ No labels from the target domain
- ✓ No access to the target domain during training
- ✓ Adapts using only the unlabelled test batch itself

TTA operates entirely at inference time, making it compatible with any
pre-trained model and applicable to new deployment scenarios without
retraining.

---

## TTA Protocol

Each method is evaluated independently on each (corruption type, severity)
pair following the same protocol:

```
for each corruption_type in {gaussian_noise, ..., jpeg_compression}:
    for each method in {no_adaptation, TTN, TENT, pseudo_label}:
        1. RESET: restore model to pre-adaptation state
        2. ADAPT: run method-specific forward (and optionally backward) passes
        3. EVALUATE: record top-1 accuracy on the full test batch (10 000 samples)
        4. RECORD: store accuracy, entropy_before, entropy_after
```

**Critical design decision — resetting between corruptions:**
After each corruption type, the model is fully restored to its original
pre-adaptation weights (via `model.restore_original_state()`).  This prevents
*cross-contamination*: adaptation learned for `gaussian_noise` must not
artificially inflate the score for `fog`.  Without reset, accumulated BN
statistics from one corruption bias the evaluation of subsequent ones.

---

## Dataset: CIFAR-10-C

| Property       | Value                                 |
|----------------|---------------------------------------|
| Source         | Hendrycks & Dietterich (2019)         |
| Base dataset   | CIFAR-10 (clean test set)             |
| Corruptions    | 15 types × 5 severities               |
| Images/split   | 10 000 per (corruption, severity)     |
| Image size     | 32 × 32 × 3 (uint8)                  |
| Classes        | 10 (airplane, car, bird, …, truck)    |
| Download       | zenodo.org/record/2535967 (~300 MB)   |

### Corruption Categories

| Category | Corruptions |
|----------|-------------|
| Noise    | gaussian_noise, shot_noise, impulse_noise |
| Blur     | defocus_blur, glass_blur, motion_blur, zoom_blur |
| Weather  | snow, frost, fog, brightness |
| Digital  | contrast, elastic_transform, pixelate, jpeg_compression |

---

## TTA Methods

### Method 1 — No Adaptation (Baseline)

Direct inference on corrupted data using the source-domain model.  Batch
normalisation layers use the running statistics estimated on clean CIFAR-10
training data, which are mismatched to the corrupted test distribution.  This
establishes the performance floor that all TTA methods must beat.

**Computational cost:** 1 forward pass per batch.

---

### Method 2 — Test-Time Normalisation (TTN)

*Schneider et al., NeurIPS 2020*

BN layers maintain running statistics (μ, σ²) estimated on the source domain.
TTN corrects this mismatch by replacing the stored statistics with those of the
test batch:

```
for each test batch x:
    model.train()            # BN computes batch statistics
    with no_grad:
        _ = model(x)         # updates running_mean, running_var
    model.eval()
    predictions = model(x)  # uses updated statistics
```

The exponential moving average update rule is:

```
running_mean ← (1 − m) · running_mean + m · batch_mean
running_var  ← (1 − m) · running_var  + m · batch_var
```

where `m = 0.1` (default PyTorch momentum).

**Why only BN statistics?** BN normalises activations to zero mean and unit
variance.  When the source and target distributions differ, the stored μ and σ²
are incorrect, leading to biased activations.  Updating them from the test
batch corrects this without any gradient computation.

**Computational cost:** 2 forward passes per batch (no backward pass).

---

### Method 3 — TENT: Test Entropy Minimisation

*Wang et al., ICLR 2021*

TENT minimises prediction entropy using gradient descent on BN affine
parameters (γ, β) only:

```
Loss = H(p) = -Σ softmax(logits)_i · log_softmax(logits)_i

for each test batch x:
    freeze all parameters except BN γ, β
    set BN layers to training mode
    logits = model(x)
    loss   = H(logits)          # entropy loss — no labels needed
    loss.backward()
    optimizer.step()             # updates γ, β only
    predictions = model(x)      # eval mode
```

**Why only BN affine parameters?**

1. *Prevents catastrophic forgetting*: Conv and FC weights encode
   ImageNet/CIFAR-10 features.  Updating them on a small test batch
   would overwrite useful knowledge.
2. *Efficiency*: BN affine parameters (2 scalars per channel per layer)
   are far fewer than conv filter weights.  For ResNet-50: ~10 K BN affine
   params vs. ~23 M total parameters.
3. *Distribution sensitivity*: γ and β directly modulate the scale and
   shift of feature maps, making them the most efficient adaptation target
   for distribution shift.

**Why entropy as the loss?**  
No labels are available, so supervised cross-entropy is impossible.  Entropy
measures model uncertainty: minimising it encourages confident predictions,
which implicitly aligns the feature distribution to the test domain.  This is
the only self-supervised signal available without labels.

**Computational cost:** 1 forward + 1 backward pass per batch per gradient step.

---

### Method 4 — Pseudo-Label Adaptation

*Sun et al., ICML 2020*

```
for each test batch x:
    probs = softmax(model(x))
    max_probs, pseudo_labels = probs.max(dim=1)
    
    # Filter: only high-confidence predictions
    mask = max_probs >= 0.9
    if mask.sum() == 0:
        skip  # no confident predictions; fall back to baseline
    
    # Fine-tune on confident subset
    model.train()
    loss = cross_entropy(model(x[mask]), pseudo_labels[mask])
    loss.backward()
    optimizer.step()
    
    # Evaluate on full batch
    model.eval()
    predictions = model(x)
```

**Why threshold = 0.9?**  
Lower thresholds include too many incorrect pseudo-labels.  At threshold 0.9,
roughly the top 10–30% of most-confident predictions are used, providing a
good quality/quantity trade-off for CIFAR-10 at standard batch sizes.

**Known failure: Confirmation Bias on Blur Corruptions**  
Blur corruptions cause the model to assign high confidence to wrong classes.
A blurred cat image may be classified as "dog" with 97% confidence, passing
the threshold.  Fine-tuning on these wrong pseudo-labels reinforces the error
and degrades accuracy below the no-adaptation baseline.  This is the
*confirmation bias* failure mode.

---

## Evaluation Metric: Mean Corruption Error (mCE)

Following Hendrycks & Dietterich (2019), we measure:

```
Error_c  = 1 − Accuracy_c              (corruption error for corruption c)
mCE      = (1/|C|) Σ_{c ∈ C} Error_c  (mean over all C corruptions)
```

Lower mCE is better.  The relative improvement over the baseline:

```
RelImp = (mCE_baseline − mCE_method) / mCE_baseline
```

A positive RelImp indicates the method reduces corruption error compared
to the no-adaptation baseline.

*Note:* The original Hendrycks & Dietterich formulation normalises by
AlexNet's corruption error.  Our simplified mCE is the raw mean error rate,
which is more interpretable and requires no AlexNet baseline model.

---

## Entropy–Accuracy Correlation Method (RQ3)

**Hypothesis:** Pre-adaptation entropy H̄_pre measures how "surprised" the
model is by a corruption.  High surprise → large distribution shift →
TTA should help more.

**Operationalisation:**

1. For each corruption type *c* at severity 3:
   - Compute H̄_pre(*c*) = mean Shannon entropy over 10 000 test samples
     using the unmodified model.
   - Run TENT; record ΔAcc(*c*) = Acc_TENT(*c*) − Acc_baseline(*c*).

2. Compute Pearson correlation:
   ```
   r = Σ((H_c − H̄)(ΔAcc_c − ΔĀcc)) / sqrt(Σ(H_c − H̄)² · Σ(ΔAcc_c − ΔĀcc)²)
   ```
   where the sums are over all 15 corruption types.

**Interpretation:**
- r ≈ +1: High-entropy corruptions benefit most from TENT (hypothesis supported).
- r ≈ 0: Entropy does not predict TENT benefit.
- r ≈ −1: Unexpected — high-entropy corruptions benefit *less* from TENT.

**Expected finding:** Noise corruptions produce higher H̄_pre than digital
corruptions, and they also show larger TENT gains, giving r > 0.

---

## Model Architecture

**ResNet-50** (He et al., 2016) with CIFAR-10 adaptations:

| Component | Standard ResNet-50 | CIFAR-10 Adaptation |
|-----------|--------------------|---------------------|
| conv1     | 7×7, stride 2      | 3×3, stride 1       |
| maxpool   | 3×3, stride 2      | removed (Identity)  |
| fc        | 2048 → 1000        | 2048 → 10           |
| BN layers | 53 BatchNorm2d     | unchanged           |

Fine-tuning protocol: only `layer4`, FC, and BN affine parameters are updated
for 10 epochs on CIFAR-10 training data (SGD, lr=0.01, cosine schedule).
Earlier backbone layers retain ImageNet features.

---

## References

1. Hendrycks D., Dietterich T. (2019). "Benchmarking neural network robustness
   to common corruptions and perturbations." ICLR 2019.
   https://arxiv.org/abs/1903.12261

2. Wang D., Shelhamer E., Liu S., Olshausen B., Darrell T. (2021).
   "Tent: Fully test-time adaptation by entropy minimization." ICLR 2021.
   https://arxiv.org/abs/2006.10726

3. Schneider S., Rusak E., Eck L., Bringmann O., Brendel W., Bethge M. (2020).
   "Improving robustness against common corruptions by covariate shift
   adaptation." NeurIPS 2020.
   https://arxiv.org/abs/2006.16971

4. Sun Y., Wang X., Liu Z., Miller J., Efros A., Hardt M. (2020).
   "Test-time training with self-supervision for generalization under
   distribution shifts." ICML 2020.
   https://arxiv.org/abs/1909.13231

5. He K., Zhang X., Ren S., Sun J. (2016).
   "Deep residual learning for image recognition." CVPR 2016.
   https://arxiv.org/abs/1512.03385
