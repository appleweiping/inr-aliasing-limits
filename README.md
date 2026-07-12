# Silent Aliasing in Fixed Fourier-Feature Coordinate Models

Identifiability, sampling design, and detection for coordinate models built on a
**fixed finite set of Fourier features** — the linear core of Fourier-feature implicit
neural representations. Target venue: IEEE ICASSP (Signal Processing Theory & Methods).

**Scope, stated up front.** Every theorem here is about the fixed-feature,
linear-coefficient model (least squares on an exponential dictionary Λ). Trained
nonlinear networks (Fourier-feature MLPs, SIRENs) appear **only as empirical
extrapolation**, with predictions from each network's own NTK linearization and all
seed-level failures reported. Adaptive / learned frequency sets are **outside** the
theory. This is not a theory of "all INRs".

## The four results (each Monte-Carlo-pinned in `tests/`)

* **T1 — exact equivalence and silent aliasing** (`identifiability.py`): on *any* subset
  of a rate-Q sampling grid, an integer tone with ν ≡ ω (mod Q) is **exactly**
  indistinguishable from the model atom ω: zero training residual, full-energy
  reconstruction failure, and — by a two-point argument — *no estimator whatsoever*
  can avoid error ≥ |a|/√2. Visibility `v_T(ν)` and aliasability `a_T(ν)` classify the
  general case; classical mod-N aliasing is the full-grid special case.
* **T2 — function-space accounting**: closed-form continuous Gram matrix + Riesz bounds
  convert coefficient errors into signal errors, and an exact decomposition keeps
  **aliasing-induced modeled-component error** and **truncation error** separate —
  they are different things and only the former responds to sampling design.
* **T3 — random sampling breaks exact aliasing**: under timing jitter the fold's
  visibility follows the jitter characteristic function (Gaussian: `v ≈ 2π|r|Q·σ_t`);
  under i.i.d. sampling the worst-case aliasability over a finite candidate set is
  ≤ `√m·ε/(λ_min − m·ε)` with `ε = 2√(log(4(mK+m²)/δ)/N)`, with probability ≥ 1−δ
  (proved; validated with the bound overlaid on the empirics).
* **T4 — detection dichotomy**: exactly coherent folds are undetectable by **any**
  sample-only test (identical observation distributions); visible tones admit exact
  noncentral-χ² detection power, matched empirically by five calibrated sample-only
  detectors (calibration / test sets strictly separated).

Full proofs: [`paper/supplement.pdf`](paper/supplement.tex). Prior-art audit:
[`docs/novelty-matrix.md`](docs/novelty-matrix.md) (what is known, what we state, what
is strictly new — old "Theorem 1" is demoted to a background lemma; the average-case
decomposition is background Prop. S1 in the supplement).

## Experiments (theory ↔ experiment closed loop)

| Experiment | Script | What it validates |
|---|---|---|
| E1 synthetic matrix | `experiments/run_synthetic_matrix.py` | T1 grid persistence vs i.i.d. (fixed *and* adversarial tones), T3 jitter law + concentration bound, T2 decomposition |
| E2 detection | `experiments/run_diagnostic_roc.py` | T4 power curve vs 5 sample-only detectors; calibration/test separation, bootstrap CIs, PR-AUC, TPR@1%/5% FPR; the coherent fold reported separately as the impossibility demo |
| E3 real signals | `experiments/run_real_signal.py` | **Sample-only** bandwidth selection (Lomb–Scargle, correlation periodogram, CV, fixed, ridge) vs an oracle that is used for evaluation only; anti-aliased `resample_poly` decimation; 9 speech segments from 3 recordings; paired 95% CIs |
| E4 trained nets | `experiments/run_nonlinear.py` | 20 sampling seeds × 3 tones × 3 architectures; ablation-measured fold vs **NTK-linearization** prediction; label-permutation / amplitude / weight-seed controls; failures reported |
| E5 2-D | `experiments/run_image2d.py` | Part A: exact 2-D frequency-vector fold on the linear model. Part B: lattice (coherent) vs random masks on 3 images — predicted DFT replicas appear only under lattice masks; ridge/early-stop baselines; clean/noisy/held-out/full-field PSNR reported separately |

Every result JSON carries a `_meta` stamp (git commit, seeds in config, Python/NumPy/
SciPy/torch/CUDA/GPU versions).

## Reproduction

```bash
uv venv --python 3.11 && source .venv/bin/activate   # Windows: .venv/Scripts/activate
uv pip install -r requirements.lock && uv pip install -e . --no-deps
python -m pytest -q                     # theorem <-> Monte Carlo suite (CPU)
python experiments/run_all.py           # CPU-only reproduction (E1-E3 + E5 part A)
python experiments/run_all.py --full    # + trained-network studies (needs torch + GPU)
```

`requirements.lock` pins the CPU stack. Torch experiments were run with torch 2.8.0
+cu128 on an RTX 4090 (recorded per-JSON in `_meta`); GPU training sets
`torch.use_deterministic_algorithms(True, warn_only=True)`, but bitwise reproducibility
across CUDA/toolkit versions is not guaranteed — expect statistically, not bitwise,
identical results there. The CPU pipeline is deterministic (fixed seeds).

To build the paper and supplement:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && pdflatex supplement
```

## Data provenance & licenses

| Data | Source | License / terms | Processing |
|---|---|---|---|
| Speech (3 recordings) | Open Speech Repository, Harvard sentences (`OSR_us_000_00{10,11,12}_8k.wav`) | Free for research use per OSR site; **not** covered by this repo's MIT license | SHA-256 checksums + per-file metadata in `data/speech_provenance.json`; 2-s segments; `resample_poly` anti-aliased decimation |
| Mauna Loa CO₂ | NOAA GML (`co2_mm_mlo.csv`) | Public (NOAA data disclaimer) | monthly means, full record, linear detrend |
| Sunspots (daily + 13-month smoothed) | SILSO, Royal Observatory of Belgium | CC BY-NC 4.0 — check before commercial reuse | the *smoothed* series is an intentionally low-pass **proxy** and is labelled as such everywhere |

Small `.npz` copies are vendored so experiments reproduce offline; the third-party data
remain under their own terms (the MIT license below covers the **code only**).

## Repository layout

```
src/inralias/     identifiability.py (T1-T4 core)  sampling.py  limits.py (background)
                  inr.py  diagnostics.py  signals.py
experiments/      run_synthetic_matrix.py  run_diagnostic_roc.py  run_real_signal.py
                  run_nonlinear.py  run_image2d.py  run_all.py
tests/            test_identifiability.py  test_theory_vs_sim.py  test_sampling.py
                  test_diagnostics_inr.py          (34 tests, CI on every push)
paper/            main.tex + supplement.tex (full proofs) + figures + refs.bib
docs/             novelty-matrix.md (prior-art audit)
results/          *.json (with _meta provenance) + figures/
```

## Honesty notes

* The concentration bound is a union bound: valid, correct rate, loose constants — it is
  plotted against the empirics rather than replaced by them.
* Cross-validated bandwidth selection is reported even where it *underperforms* the
  periodogram selectors (it is fragile under gappy sampling).
* Trained-network fold agreement is reported as a distribution over ≥20 seeds including
  disagreements; nothing is filtered.
* The diagnostic's fixed decision threshold must come from `null_calibrated_threshold`
  (explicit null calibration); no a-priori threshold is claimed as validated.
* "Monte-Carlo-pinned" means the closed forms are asserted against independent
  simulation with tight tolerances in `tests/` — it is verification of the formulas'
  implementations, not a substitute for the proofs (which are in the supplement).

## AI assistance disclosure

Substantial portions of the code, experiments, and manuscript drafts in this repository
were produced with AI assistance (Claude, Anthropic), under human direction, with all
results generated by the released code. Git history records this via `Co-Authored-By`
trailers and has not been rewritten. Venue-specific AI-use disclosure will follow the
policy of the submission year.

## 中文速览

研究对象是**固定傅里叶特征坐标模型**（有限频率集 Λ + 线性系数——傅里叶特征 INR 的线性核心）。
四个定理：T1 在任意 rate-Q 采样网格子集上，ν ≡ ω (mod Q) 的带外音调与模型原子**精确不可
辨识**（训练残差为零、重建误差不小于全部带外能量、任何估计器都无法避免——两点法下界）；
T2 连续域 Gram/Riesz 把系数误差换算为信号误差，并把"混叠偏差"与"截断误差"严格分开；
T3 随机采样**可证地破坏**精确混叠——抖动下可见性等于抖动特征函数（高斯：v ≈ 2π|r|Qσ_t），
iid 采样下有限候选集上最坏混叠度以显式速率集中到零；T4 检测二分性——相干折叠对任何仅用
样本的检测器都不可检测，可见音调的检测功率有精确的非中心 χ² 公式，并被五种校准检测器
实证复现。训练的非线性网络只作为经验外推（用各网络自己的 NTK 线性化做预测，全部种子级
失败如实报告）；自适应频率不在理论覆盖范围内。

## License

MIT (code only; see `LICENSE`). Third-party data under their own terms (table above).
