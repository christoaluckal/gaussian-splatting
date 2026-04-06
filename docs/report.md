# Gaussian Splatting Experiment Report

## 1. Experiment Summary

- **Baseline:** Standard Gaussian Splatting initialization
- **Proposed:** Smarter point and image initialization
- **Goal:** Improve reconstruction quality while keeping memory, model size, and runtime increases reasonable
- **Main question:** Does the proposed initialization improve quality because it is better initialized, or only because it ends with more Gaussians?

---

## 2. Experimental Setup

### 2.1 Data
- **Dataset / scene(s):**
- **Number of input views:**
- **Image resolution:**
- **Train / test split:**

### 2.2 Training Configuration
- **Total iterations:**
- **Densification settings:**
- **Pruning settings:**
- **Opacity reset schedule:**
- **SH degree / feature settings:**
- **Optimizer settings:**
- **Random seed(s):**

### 2.3 System Configuration
- **GPU:**
- **CPU:**
- **RAM:**
- **PyTorch version:**
- **CUDA version:**
- **OS:**

### 2.4 Measurement Protocol
- **Quality metrics:** PSNR, SSIM, LPIPS
- **Representation metrics:** final number of Gaussians, Gaussian model memory (MB)
- **Runtime metrics:** peak allocated GPU memory (MB), peak reserved GPU memory (MB), total training time (s), average iteration time (ms)
- **Evaluation checkpoints:**
- **Peak memory reset point:** before full run / before training loop

---

## 3. Metrics Collected

### 3.1 Quality
- **PSNR**
- **SSIM**
- **LPIPS**

### 3.2 Representation Size
- **Final number of Gaussians**
- **Gaussian model memory (MB)**

### 3.3 Runtime / System Cost
- **Peak allocated GPU memory (MB)**
- **Peak reserved GPU memory (MB)**
- **Total training time (s)**
- **Average iteration time (ms)**

### 3.4 Efficiency
- **PSNR gain over baseline**
- **PSNR per 100k Gaussians**
- **PSNR per GB peak allocated memory**
- **Time to reach target PSNR**
- **PSNR at matched budget**

---

## 4. Main Final Comparison

| Method | PSNR | SSIM | LPIPS | Final #Gaussians | Gaussian Model MB | Peak Allocated MB | Peak Reserved MB | Total Train Time (s) | Avg Iter Time (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |  |  |  |
| Delta |  |  |  |  |  |  |  |  |  |

### 4.1 Notes on Final Comparison
- Did PSNR improve substantially?
- Did Gaussian count increase moderately or heavily?
- Did Gaussian model memory scale roughly with Gaussian count?
- Did peak allocated memory remain reasonable?
- Did training time stay comparable?

---

## 5. Budget-Matched Comparisons

This section is critical for showing that the improvement is not only due to a larger final model.

### 5.1 Matched Number of Gaussians

| Budget Type | Baseline PSNR | Proposed PSNR | Baseline SSIM | Proposed SSIM | Baseline LPIPS | Proposed LPIPS | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Same #Gaussians |  |  |  |  |  |  |  |

### 5.2 Matched Gaussian Model Memory

| Budget Type | Baseline PSNR | Proposed PSNR | Baseline SSIM | Proposed SSIM | Baseline LPIPS | Proposed LPIPS | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Same Gaussian MB |  |  |  |  |  |  |  |

### 5.3 Matched Peak Allocated GPU Memory

| Budget Type | Baseline PSNR | Proposed PSNR | Baseline SSIM | Proposed SSIM | Baseline LPIPS | Proposed LPIPS | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Same Peak Alloc MB |  |  |  |  |  |  |  |

### 5.4 Matched Training Time

| Budget Type | Baseline PSNR | Proposed PSNR | Baseline SSIM | Proposed SSIM | Baseline LPIPS | Proposed LPIPS | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Same Train Time |  |  |  |  |  |  |  |

### 5.5 Budget-Matched Interpretation
- Does the proposed method still outperform the baseline at equal budget?
- If yes, this supports the claim that initialization quality improved efficiency.
- If not, the method may still be useful, but the claim becomes "better quality for higher cost" rather than "better efficiency."

---

## 6. Convergence Analysis

### 6.1 PSNR vs Iteration
- **Observation:**
- **Key takeaway:**

### 6.2 PSNR vs Wall-Clock Time
- **Observation:**
- **Key takeaway:**

### 6.3 Number of Gaussians vs Iteration
- **Observation:**
- **Key takeaway:**

### 6.4 Gaussian Model Memory vs Iteration
- **Observation:**
- **Key takeaway:**

### 6.5 Peak Allocated Memory Over Training
- **Observation:**
- **Key takeaway:**

### 6.6 Convergence Interpretation
Use this section to answer:
- Does the proposed method start better?
- Does it converge faster?
- Does it end at a better final solution?
- Does it need substantially more memory or time to do so?

---

## 7. Derived Efficiency Metrics

| Metric | Baseline | Proposed | Delta |
|---|---:|---:|---:|
| PSNR per 100k Gaussians |  |  |  |
| PSNR per GB Peak Allocated |  |  |  |
| Time to Reach Target PSNR |  |  |  |
| Gaussian Model MB per 100k Gaussians |  |  |  |

### 7.1 Interpretation of Efficiency Metrics
- Higher PSNR per budget is better
- Lower time to target PSNR is better
- If the proposed method uses more Gaussians but still shows better PSNR per memory budget, that is strong evidence of improved efficiency

---

## 8. Scene-by-Scene Breakdown

Use this section if multiple scenes are evaluated.

| Scene | Method | PSNR | SSIM | LPIPS | Final #Gaussians | Gaussian Model MB | Peak Allocated MB | Train Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Scene 1 | Baseline |  |  |  |  |  |  |  |
| Scene 1 | Proposed |  |  |  |  |  |  |  |
| Scene 2 | Baseline |  |  |  |  |  |  |  |
| Scene 2 | Proposed |  |  |  |  |  |  |  |

### 8.1 Cross-Scene Summary
- **Average PSNR gain:**
- **Average Gaussian count increase:**
- **Average Gaussian model memory increase:**
- **Average peak allocated memory increase:**
- **Average training time increase:**

---

## 9. Failure Cases and Caveats

List any important limitations here.

- Scene(s) where improvement was small:
- Scene(s) where memory overhead was unusually high:
- Scene(s) where convergence was unstable:
- Cases where matched-budget advantage disappeared:
- Cases where the proposed method needed noticeably more time:

### 9.1 Caveat Interpretation
This section helps separate:
- true robustness
- scene-specific gains
- gains caused mainly by extra capacity

---

## 10. Main Interpretation

Write 1 to 3 paragraphs here.

Suggested structure:

1. State whether the proposed method improved quality.
2. State whether the cost increase was moderate or large.
3. State whether matched-budget comparisons support a true efficiency gain.

Example wording:

The proposed initialization improves reconstruction quality over the baseline, as shown by higher PSNR and supporting perceptual metrics. While it increases the final number of Gaussians and representation memory, the increase is moderate relative to the quality gain.

Peak allocated GPU memory and training time also increase, but the proposed method remains favorable if the quality gain is substantially larger than the cost increase. If the matched-budget comparisons still favor the proposed method, this suggests that the improvement comes from better initialization rather than simply increasing model capacity.

---

## 11. Conclusion

Write 2 to 4 sentences here.

Suggested template:

The proposed initialization improves reconstruction quality by **[X] dB PSNR** relative to the baseline. This comes with a **[Y]%** increase in final Gaussian count, a **[Z]%** increase in Gaussian model memory, and a **[W]%** increase in peak allocated GPU memory. If the proposed method also outperforms the baseline at matched Gaussian count, memory, or training-time budgets, then the results support the conclusion that smarter initialization improves optimization efficiency rather than only increasing final model size.

---

## 12. Recommended Figures

Include the following plots in the report if available:

- PSNR vs iteration
- PSNR vs wall-clock time
- number of Gaussians vs iteration
- Gaussian model memory vs iteration
- peak allocated GPU memory over training
- scene-by-scene PSNR bar chart
- matched-budget comparison chart

---

## 13. Appendix: Raw Values

### 13.1 Baseline Raw Run Metadata
- Run ID:
- Checkpoint:
- Seed:
- Notes:

### 13.2 Proposed Raw Run Metadata
- Run ID:
- Checkpoint:
- Seed:
- Notes:

### 13.3 Raw CSV / Log Paths
- Baseline metrics CSV:
- Proposed metrics CSV:
- Evaluation summary CSV:
- Plot directory: