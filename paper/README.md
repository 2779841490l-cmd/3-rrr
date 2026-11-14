# Paper Assets & Ground-Truth Index (`paper/`)

> **Title**: *Sub-100-ms, 20-$\mu$m Point-to-Point Positioning of a High-Backlash 3-RRR Parallel Mechanism via Sub-Microsecond Ballistic Control and In-Flight Switching*  
> **Author**: Qiuhan Liu (School of Mechanical and Electronic Engineering, Wuhan University of Technology, Wuhan 430070, China)  
> **Target Venues**: *IEEE Transactions on Instrumentation and Measurement (IEEE TIM)* / *IEEE/ASME Transactions on Mechatronics (IEEE T-MECH)*  
> **Frozen Hardware/Code Baseline**: Commit `45ea2aa`  
> **Consolidated Paper & LaTeX Deliverable**: Commit `be8b8b2`

---

## 1. Deliverable Files

| File / Folder | Purpose | Status |
| :--- | :--- | :--- |
| `paper/manuscript.md` | Publication-clean Markdown manuscript (zero code/CSV/hash artifacts, de-AI IEEE style) | Finalized (`be8b8b2`) |
| `paper/latex/main.tex` | Complete `\documentclass[journal]{IEEEtran}` Overleaf source (`booktabs`, `algorithm`, `IEEEproof`) | Ready for Overleaf |
| `paper/latex/references.bib` | 15 search-verified BibTeX references (`[1]`–`[15]`, zero unverified entries) | Synced with `paper/references.bib` |
| `paper/latex/figures/` | Clean 300-DPI PNG figures matching Figs. 2–5 in `main.tex` | Ready |
| `paper/paper_overleaf_submission.zip` | Direct Overleaf upload archive (`main.tex`, `references.bib`, `figures/*.png`) | Packaged (`1.52 MB`) |
| `plot_paper_figures.py` | Deterministic generator producing clean 300-DPI Figs. 2–5 from frozen CSVs | Clean labels verified |
| `paper/PAPER_OUTLINE.md` | Early pre-redesign brainstorm outline (superseded; retained only for history) | Archived |

> **One Remaining Manual Action Before Journal Submission**: Capture and label an overhead hardware photograph and place it at `paper/latex/figures/fig0_hardware_photo.jpg` (Fig. 1 currently renders a safe `\IfFileExists` framed placeholder box so Overleaf compiles cleanly without errors).

---

## 2. Figure Mapping (Manuscript & LaTeX)

| Paper Fig. # | File Name (under `figures/`) | Source Data (`experiments/`) | Key Findings Shown |
| :---: | :--- | :--- | :--- |
| **Fig. 1** | `fig0_hardware_photo.jpg` *(author upload)* | Physical testbed photo | Dual-Y 3-RRR stage & 3200-DPI optical sensor layout |
| **Fig. 2** | `fig1_deadzone_3regime.png` | `deadzone_gain_fit.csv`, `step_scan.csv` | Deadzone plateau ($\le 15\ \mu\text{s}$), breakaway ($\pm 20\ \mu\text{s}$), Stribeck slide, $8.15 \pm 5.17\ \text{cnt}$ stick-slip jumps |
| **Fig. 3** | `fig2_short_arm_jacobian.png` | `real_jacobian_2x2.csv`, `real_jacobian_2x2_short_arm.csv` | $-49.6\%$ $u_2$ gain compression ($4.332 \to 2.182\ \text{cnt}/\mu\text{s}$), $K_+$ vs. $K_-$ asymmetry |
| **Fig. 4** | `fig3_ablation_evolution.png` | `ablation_evolution.csv`, `short_ablation_trials.csv` | 4-stage architecture progression (`41.95 -> 2.39 cnt`) & `NO_PRELOAD` / `STAGE1_ONLY` rollbacks |
| **Fig. 5** | `fig4_gate1_gate2a_results.png` | `gate1_30_20_eval.csv`, `gate2_6dir_eval.csv` | Protocol I 12-shot convergence (`1.83 cnt`) & Protocol II 6-dir residuals (`2.39 cnt`, `98.2 ms`) |

---

## 3. Verified Quantitative Results (`1 count = 7.9375 µm` at 3200 DPI)

- **Workspace (SOCP Chebyshev Manifold)**: Semi-spans $390.0 \times 300.0\ \text{counts}$ ($3.096 \times 2.381\ \text{mm}$); bilateral full span $\pm 390.0 \times \pm 300.0\ \text{counts}$ ($780.0 \times 600.0\ \text{counts}$, $6.191 \times 4.763\ \text{mm}$); offset $[150.0, 220.0]^T$; equilibrium $\mathbf{u}_0^* = [1496.0, 1466.0]^T\ \mu\text{s}$ with $u_{y1} + u_{y2} = 3000\ \mu\text{s}$.
- **Short-Arm Gain & Compliance**: $J_{\text{short}} = \begin{bmatrix} +2.0222 & -1.6941 \\ -0.2595 & -1.3760 \end{bmatrix}\ \text{cnt}/\mu\text{s}$; sensitivity $|du_1| = 2.039\ \text{cnt}/\mu\text{s}$, $|du_2| = 2.182\ \text{cnt}/\mu\text{s}$ (compressed $-49.6\%$ from $4.332\ \text{cnt}/\mu\text{s}$); deadzones $d_{u1} = 9.22\ \mu\text{s}$ ($+:10.51, -:7.93$), $d_{u2} = 4.00\ \mu\text{s}$ ($+:3.04, -:4.96$).
- **Protocol I (Gate 1, `(30, 20)` $\times$ 12 shots)**: All-12 mean $2.88\ \text{cnt}$ ($t_{\text{bal}} = 84.4\ \text{ms}$, $t_{\text{set}} = 116.1\ \text{ms}$); **Last-5 Trimmed-4 Mean = $1.83\ \text{cnt}$ ($14.5\ \mu\text{m}$)**, median $2.24\ \text{cnt}$, raw5 mean $2.18\ \text{cnt}$, raw max $3.61\ \text{cnt}$, $t_{\text{bal}} = 86.7\ \text{ms}$, $t_{\text{set}} = 118.6\ \text{ms}$ (`PASS`).
- **Protocol II (Gate 2a, 6 directions $\times$ 5 shots, zero ILC)**: **6-Dir Trimmed-3 Mean = $2.39\ \text{cnt}$ ($19.0\ \mu\text{m}$)**, median $3.16\ \text{cnt}$, **$t_{\text{bal}} = 83.6\ \text{ms}$**, **$t_{\text{set}} = 98.2\ \text{ms}$** (`5/6 PASS`, overall `GATE PASS`; `(+30, +25)` at $3.61\ \text{cnt}$ exceeds $3.5$ by $0.11\ \text{cnt}$; `(0, -35)` best at $0.80\ \text{cnt} / 6.4\ \mu\text{m}$).
- **Comparisons & Ablations**:
  - Continuous PID: $101.04\ \text{cnt}$ ($802.0\ \mu\text{m}$), $1244.9\ \text{ms}$ $\to$ IF-OPB gives **$97.63\%$ error reduction, $12.68\times$ speedup**.
  - Stage Evolution (`① -> ② -> ③ -> ④`): `41.95 cnt (229.9 ms)` $\to$ `5.67 cnt (139.3 ms)` $\to$ `4.17 cnt (125.4 ms)` $\to$ **`2.39 cnt (98.2 ms)`**.
  - Rollbacks (`(30,20)` / `(0,+35)` Trim-3): `NO_PRELOAD = 5.20 / 4.76 cnt`; `STAGE1_ONLY = 10.28 / 5.13 cnt`.

---

## 4. Architecture Decisions & Falsified Early Ideas (Do Not Re-Introduce)

1. **PWM Carrier = 100 Hz ($T_{\text{pwm}} = 10000\ \mu\text{s}$), NOT 200 Hz**: At $200\ \text{Hz}$ ($5.0\ \text{ms}$ period) and $1.85\ \text{ms}$ max pulse, low recovery is $5.0 - 1.85 = 3.15\ \text{ms} < T_{\text{low,min}} = 4500\ \mu\text{s}$, depleting servo timing ramp capacitors.
2. **Rigid Dual-Y Constraint ($u_{y1} + u_{y2} = 3000\ \mu\text{s}$)**: Micro-yaw differential compensation was discarded; pure complementarity avoids opposing-gear binding.
3. **Pure Time-Gated Position Cutoff ($p_\parallel \ge r_{\text{cut}} L$ with $t_{\min} \in \{36, 52\}\ \text{ms}$)**: Velocity differentiation ($v = \Delta p / \Delta t$) over $6\ \text{ms}$ intervals amplified quantization jitter into early $65\text{--}72\ \text{ms}$ false cutoffs and was permanently removed.
4. **Frozen Cross-Shot ILC for Short Chords ($L \le 70\ \text{cnt}$)**: Prevents Stage-1 cross-axis coupling from fighting Stage-2 trim.
5. **Stage-2 Guardrails**: Cartesian no-reversal clamp ($e_i \cdot (p_{\text{target},i} - p_{\text{start},i}) \le 0 \Rightarrow e_i \gets 0$), $B\to\text{eq}$ retraction $Y$-lock when $|\Delta u_{\text{bst},2} - \Delta u_{\text{eq},2}| > 10\ \mu\text{s}$, no-harm stop gate ($\|\mathbf{e}_1\|_2 \le 2.8\ \text{cnt}$), and descent acceptance margin $\|\mathbf{e}_1 - J_{\text{short}}\Delta\mathbf{u}_{S2}\|_2 < \|\mathbf{e}_1\|_2 - 0.30$.
