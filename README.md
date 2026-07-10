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
> real signal?** An INR can only represent frequencies in a *structured* representable set
> `Λ` fixed by its features and width/depth. We show that (1) if the signal's spectrum lies
> in `Λ` and the (possibly nonuniform, noisy) samples form a frame for `Λ`, a
> bandwidth-matched INR recovers it with a closed-form **noise gain** `κ = 1/σ_min(Φ)`
> (*achievability*); and (2) any out-of-band energy is **folded** onto `Λ` with a sharp
> **aliasing error floor** that no width-`W` INR can beat at that sampling density
> (*converse*) — and, strikingly, the global reconstruction error can look fine while the
> recovered **spectrum** is silently corrupted (**silent aliasing**). A sharp phase
> transition separates the two regimes, we give a matched estimator and a **ground-truth-free
> aliasing diagnostic**, and we instantiate **both sides of the limit** on real audio,
> scientific (astronomical / seismic) and RF signals.

## Status

- [x] Sampling / frame machinery on a finite frequency dictionary (`src/inralias/sampling.py`, tested)
- [x] Theorem 1 achievability — noise gain, recoverable MMSE (validated vs Monte-Carlo)
- [x] Theorem 2 converse — aliasing bias, error floor, folded frequency, silent aliasing (validated)
- [x] Theorem 3 statistical — aliasing-bias floor vs estimation variance (validated)
- [x] Trained nonlinear INRs (SIREN / Fourier-feature MLP) + bandwidth-matched estimator + data-only diagnostic
- [x] E1 synthetic phase transition, E2 silent-aliasing spectrum
- [x] Real data: audio (speech) and science (CO₂, sunspots) — both sides of the limit
- [x] Nonlinear-INR persistence + 2-D image demo (learned aliasing generalizes)
- [x] Paper (IEEE ICASSP, spconf) — `paper/main.pdf`, compiles clean

All theorem ↔ Monte-Carlo checks pass (`python -m pytest -q`, 17 tests).

## Why this is not "just Nyquist"

An INR's representable set `Λ` is **not** a lowpass interval: it is a *structured* frequency
dictionary (integer combinations of the network's feature/base frequencies, growing with
width/depth; Yüce et al. 2021), and for SIREN / adaptive-frequency nets `Λ` is
**signal-adaptive**. Aliasing therefore folds out-of-band energy onto a *structured, learned*
set rather than onto baseband. The classical uniform-lowpass folding law is recovered as a
special case (a correctness check in the test suite), but the general phenomena — structured
/ learned aliasing, silent aliasing, a data-only diagnostic, and bandwidth-matched
achievability — are new.

## Repository layout

```
src/inralias/     sampling.py  limits.py  inr.py  diagnostics.py  signals.py
experiments/      run_*.py + run_all.py   (regenerate every result and figure)
tests/            test_sampling.py  test_theory_vs_sim.py   (theory <-> Monte-Carlo)
paper/            main.tex (spconf) + figures + refs.bib
results/          *.json summaries + figures/
```

## Installation & reproduction

```bash
uv venv --python 3.11 && source .venv/Scripts/activate   # (or .venv/bin/activate)
uv pip install -e ".[dev]"
python -m pytest -q                # validate every theorem against Monte-Carlo (17 tests)
python experiments/run_all.py      # regenerate all results/figures (CPU parts)
```

The theory core and the synthetic + real-data experiments (E1, E2, learned-Nyquist sweeps) are
**pure numpy/scipy and run on CPU**. The trained-nonlinear-INR persistence and the 2-D image demo
need `torch` (`uv pip install ".[torch]"`; on a China network use
`--index-url https://pypi.tuna.tsinghua.edu.cn/simple`) and a GPU is convenient but not required;
`run_all.py` skips them automatically if torch is absent. To build the paper:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Real signals are fetched by `data/fetch_real.py` (public sources; see below) and small copies are
vendored under `data/*.npz` so the experiments reproduce even offline.

## Relation to prior work

- Najaf et al., *Towards a Sampling Theory for Implicit Neural Representations*
  (arXiv:2405.18410) and the super-resolution follow-up (arXiv:2506.09949) give the
  **achievability** side for *realizable, noiseless, full-Fourier* recovery. We give the
  **converse / aliasing** side under the real signal-processing setting (misspecified,
  noisy, nonuniform, time-domain) plus real data.
- The generalized-sampling / stable-sampling-rate machinery (Adcock–Hansen) and the
  INR-as-structured-dictionary view (Yüce et al. 2021) are the foundations we build on.

## 中文速览

隐式神经表示（INR，如 SIREN / 傅里叶特征网络）越来越多地被当作**连续信号模型**。现有工作
多给**架构与技巧**，最近才有人给出*可达性*采样定理（一个本身就是 INR 的信号需要多少样本才能
恢复）。本项目回答那些工作跳过的问题：**那些不是 INR 的信号（也就是所有真实信号）会怎样？**
一个 INR 只能表示其结构化可表示频率集 `Λ` 内的频率。我们证明：(1) 若信号频谱落在 `Λ` 内且
（可非均匀、含噪的）采样对 `Λ` 构成框架，带宽匹配的 INR 以闭式**噪声增益** `κ = 1/σ_min(Φ)`
恢复它（可达性）；(2) 任何带外能量都会被**折叠**到 `Λ` 上，产生一个任何宽度 `W` 的 INR 在该
采样密度下都无法突破的**混叠误差地板**（逆定理）——而且全局重建误差可能看起来正常，恢复出的
**频谱**却被悄悄污染（**静默混叠**）。一个尖锐相变把两个区制分开；我们给出匹配估计器与一个
**无需真值的混叠诊断**，并在真实音频、科学（天文/地震）与 RF 信号上展示**极限两侧**。

## Citation

Weiping Yan, *Learned Nyquist: Fundamental Aliasing Limits of Implicit Neural
Representations*, 2026. Code: `github.com/appleweiping/inr-aliasing-limits`.

## License

MIT (see `LICENSE`).
