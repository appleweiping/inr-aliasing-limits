# inr-aliasing-limits

**Learned Nyquist: Fundamental Aliasing Limits of Implicit Neural Representations.**

A complete, reproducible **AI + signal-processing-theory** research project (targeting the
IEEE ICASSP *Signal Processing Theory & Methods* track). Every theorem in the paper is
cross-checked against Monte-Carlo simulation by the test suite before it is allowed into
the manuscript.

> **TL;DR** — Implicit neural representations (INRs) — SIREN / Fourier-feature coordinate
> networks — are increasingly used as *continuous signal models*. A growing literature gives
> **architectures and tricks**; a very recent thread even gives an *achievability* sampling
> theorem (how many samples recover a signal that **is** an INR). This project answers the
> question those works skip: **what happens to the signals that are *not* INRs — i.e. every
> real signal?** A fixed-feature INR can only represent frequencies in a *structured*
> representable set `Λ` fixed by its features and width/depth. We show that (1) if the
> signal's spectrum lies in `Λ` and the (possibly nonuniform, noisy) samples form a frame for
> `Λ`, a bandwidth-matched INR recovers it with a closed-form **noise gain**
> `κ = 1/σ_min(Φ)` (*achievability*); and (2) any out-of-band energy is **folded** onto `Λ`
> with an aliasing error floor that no fixed-feature INR can beat at that sampling density
> (*converse*) — and, strikingly, for **grid-coherent** tones the sample residual is
> **exactly zero** while the reconstruction error is at least the full out-of-band energy
> (**silent aliasing**, provably invisible from the samples). A sharp phase transition at the
> stable sampling rate separates the two regimes, we give a matched estimator and a
> **ground-truth-free aliasing diagnostic with a measured ROC** (AUC 1.0 away from the
> provably-undetectable coherent worst case), and we instantiate **both sides of the limit**
> on real audio and scientific signals (speech, CO₂, sunspots) plus a 2-D image demo — the
> fold persisting, bin-for-bin (correlation 0.99 with the linear-theory bias pattern), for
> trained nonlinear and 2-D INRs.

## Status

- [x] Sampling / frame machinery on a finite frequency dictionary (`src/inralias/sampling.py`, tested)
- [x] Theorem 1 achievability — noise gain, recoverable MMSE (validated vs Monte-Carlo)
- [x] Theorem 2 converse — aliasing floor, **exact grid-fold silent aliasing** (zero residual,
      full-energy corruption at any `N ≫ m`), fold law incl. classical mod-N special case (validated)
- [x] Theorem 3 (average-case) — exact estimation-variance + aliasing-variance decomposition,
      with **every asymptotic regime pinned by tests** (coherent persistence, exact-zero,
      `O(m/N)` under i.i.d. sampling, sinc² window-leakage constant)
- [x] Ground-truth-free ring diagnostic **with measured operating characteristic**
      (`experiments/run_diagnostic_roc.py`: AUC 1.000 overdetermined/jittered; AUC 0.20 on the
      provably-undetectable coherent fold; validity guard for the underdetermined regime)
- [x] Trained nonlinear INRs (Fourier-feature MLP) — **ablation-measured fold**: the trained
      net's tone-attributable spectrum matches the linear-theory bias pattern (corr 0.99)
- [x] E1 synthetic phase transition (multi-trial, ±1 s.e., exact theory overlays), E2 exact-fold
      silent aliasing (20-seed statistics)
- [x] Real data (mean ± s.d. over 10 draws): CO₂ recovers (0.42→0.17), speech is the converse
      side (no width approaches the noise floor: truncation floor, then noise-gain blow-up)
- [x] 2-D image demo (learned aliasing persists; sample-PSNR vs full-field PSNR on-figure)
- [x] Paper (IEEE ICASSP, spconf) — `paper/main.pdf`, compiles clean (4 pages + refs-only p.5)

All theorem ↔ Monte-Carlo checks pass (`python -m pytest -q`, 22 tests).

## Why this is not "just Nyquist"

An INR's representable set `Λ` is **not** a lowpass interval: it is a *structured* frequency
dictionary (integer combinations of the network's feature/base frequencies, growing with
width/depth; Yüce et al. (CVPR 2022)). Aliasing therefore folds out-of-band energy onto a
*structured* set rather than onto baseband — for SIREN-style adaptive-frequency nets `Λ` is
additionally signal-adaptive, a case the theory defers and we probe only empirically. The
classical uniform-lowpass folding law is recovered as a special case (a correctness check in
the test suite), but the general phenomena — structured folds with non-classical targets,
exact-fold silent aliasing, a data-only diagnostic with a measured ROC, and bandwidth-matched
achievability — are new.

## Repository layout

```
src/inralias/     sampling.py  limits.py  inr.py  diagnostics.py  signals.py
experiments/      run_*.py + run_all.py   (regenerate every result and figure)
tests/            test_sampling.py  test_theory_vs_sim.py  test_diagnostics_inr.py
paper/            main.tex (spconf) + figures + refs.bib
results/          *.json summaries + figures/
```

## Installation & reproduction

```bash
uv venv --python 3.11 && source .venv/bin/activate   # (Windows: .venv/Scripts/activate)
uv pip install -e ".[dev]"
python -m pytest -q                # validate every theorem against Monte-Carlo (22 tests)
python experiments/run_all.py      # regenerate all results/figures (CPU parts)
```

The theory core and the synthetic + real-data experiments (E1, E2, the diagnostic ROC, and the
learned-Nyquist sweeps) are **pure numpy/scipy and run on CPU**. The trained-nonlinear-INR
persistence and the 2-D image demo need `torch` (`uv pip install ".[torch]"`; on a China network
use `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`) and a GPU is convenient but not
required; `run_all.py` skips them automatically if torch is absent. To build the paper:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Real signals are fetched by `data/fetch_real.py` (public sources; see below) and small copies are
vendored under `data/*.npz` so the experiments reproduce even offline.

## Relation to prior work

- Najaf & Ongie, *Towards a Sampling Theory for Implicit Neural Representations*
  (Asilomar 2024, arXiv:2405.18410) and the super-resolution follow-up (SIAM J. Imaging Sci.
  2025, arXiv:2506.09949) give the **achievability** side for *realizable, noiseless,
  full-Fourier* recovery. We give the **converse / aliasing** side under the real
  signal-processing setting (misspecified, noisy, nonuniform, time-domain) plus real data.
- **Basis mismatch** (Chi et al., IEEE TSP 2011) analyzes the same least-squares bias for a tone
  *near* a dictionary atom; our silent-aliasing theorem is the complementary **far** regime —
  a grid-coherent tone arbitrarily far from `Λ` folds with *exactly zero* residual.
- The **generalized aliasing decomposition** (Transtrum et al., arXiv:2408.08294) and the
  NTK **missing-frequency error floor** (Ma et al., arXiv:2502.05482) bound misspecification
  error for feature models; neither gives the structured fold law (*where* the energy lands),
  the exact-fold silent regime, or the detection question.
- The **spectral window** of nonuniform sampling (VanderPlas, ApJS 2018) predicts where
  out-of-band power appears in a Lomb–Scargle periodogram; our fold target is its
  dictionary-projection analogue, paired with achievability, noise, and a diagnostic.
- The generalized-sampling machinery (Adcock–Hansen) and the INR-as-structured-dictionary view
  (Yüce et al., CVPR 2022) are the foundations we build on; trained-SIREN aliasing has been
  observed empirically (arXiv:2509.09719) — we supply the theory and the diagnostic.

## 中文速览

隐式神经表示（INR，如 SIREN / 傅里叶特征网络）越来越多地被当作**连续信号模型**。现有工作
多给**架构与技巧**，最近才有人给出*可达性*采样定理（一个本身就是 INR 的信号需要多少样本才能
恢复）。本项目回答那些工作跳过的问题：**那些不是 INR 的信号（也就是所有真实信号）会怎样？**
固定特征 INR 只能表示其结构化可表示频率集 `Λ` 内的频率。我们证明：(1) 若信号频谱落在 `Λ` 内且
（可非均匀、含噪的）采样对 `Λ` 构成框架，带宽匹配的 INR 以闭式**噪声增益** `κ = 1/σ_min(Φ)`
恢复它（可达性）；(2) 任何带外能量都会被**折叠**到 `Λ` 上，产生一个任何固定特征 INR 在该
采样密度下都无法突破的**混叠误差地板**（逆定理）——而且对**网格相干**音调，样本残差**精确为
零**、重建误差却不小于全部带外能量（**静默混叠**，可证明无法从样本检测）。稳定采样率处出现
锐利相变；我们给出匹配估计器与一个**带实测 ROC 的无真值混叠诊断**（远离可证不可检测的相干
最坏情形时 AUC=1.0），并在真实音频与科学信号（语音、CO₂、太阳黑子）及一个 2-D 图像演示上展示
**极限两侧**——折叠现象在训练的非线性 INR 上逐 bin 复现（与线性理论偏差图样相关系数 0.99）。

## Citation

Weiping Yan, *Learned Nyquist: Fundamental Aliasing Limits of Implicit Neural
Representations*, 2026. Code: `github.com/appleweiping/inr-aliasing-limits`.

## License

MIT (see `LICENSE`).
