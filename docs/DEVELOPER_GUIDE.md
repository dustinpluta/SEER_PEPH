# SEER_PEPH Developer Guide and Model Summary

**Version:** 0.1.0  
**Status:** Active development — Phases 0–1 complete  
**Language:** Python 3.11+  
**Inference framework:** NumPyro (JAX backend)

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Scientific goals](#2-scientific-goals)
3. [Mathematical model specification](#3-mathematical-model-specification)
4. [Package architecture](#4-package-architecture)
5. [Module reference](#5-module-reference)
6. [Implementation roadmap](#6-implementation-roadmap)
7. [Simulation study design](#7-simulation-study-design)
8. [Development setup](#8-development-setup)
9. [Conventions and standards](#9-conventions-and-standards)
10. [Key references](#10-key-references)

---

## 1. Project overview

SEER_PEPH implements a **Bayesian spatial joint model for treatment timing and
overall survival** in colorectal cancer patients from the Georgia SEER-Medicare
cohort. The model jointly estimates:

- The effect of treatment receipt and post-treatment timing on survival
- County-level spatial frailties for treatment timing and survival
- The cross-process correlation between those two spatial frailty fields

The name reflects the core statistical machinery:
**Piecewise Exponential Proportional Hazards (PEPH)**.

This work supports two publications:

| Paper | Target journals | Primary contribution |
|---|---|---|
| Methods | *Statistics in Medicine*, *Biometrics* | Joint BYM2 spatial frailty model with cross-process correlation; identifiability analysis |
| Applied | *JNCI*, *Cancer Epidemiology*, *Medical Care* | Empirical evidence that Georgia counties with higher treatment rates have lower residual mortality |

---

## 2. Scientific goals

### Primary claim
Counties with higher-than-expected colorectal cancer treatment rates also
exhibit lower-than-expected mortality, after adjusting for patient
characteristics and area-level SES. This is quantified by the cross-process
spatial correlation parameter **ρ_u > 0** (the *healthy access hypothesis*).

### Specific aims

**Aim 1 — Treatment effect on survival**  
Quantify the effect of treatment receipt and post-treatment timing on overall
survival after adjusting for patient characteristics and geographic confounding.  
*Hypothesis:* Treatment receipt reduces mortality hazard (β_td < 0); the
survival benefit is largest in the acute post-treatment period and attenuates
over time.

**Aim 2 — Residual geographic variation**  
Characterise county-level variation in treatment timing and survival that
persists after covariate adjustment, and determine what proportion is
spatially structured (vs. area-specific noise).  
*Hypothesis:* Significant spatially structured residual variation exists in
both processes, reflecting clustering of healthcare infrastructure.

**Aim 3 — Cross-process spatial association (primary)**  
Test whether geographic variation in treatment timing and survival are
correlated across counties. Estimate the direction and magnitude of ρ_u.  
*Hypothesis:* ρ_u > 0 (healthy access hypothesis).

**Aim 4 — Methods validation**  
Demonstrate via simulation that ignoring spatial structure biases ρ_u
estimates, and characterise the conditions under which the joint spatial
model is necessary.

---

## 3. Mathematical model specification

### 3.1 Data structure

For subject $i = 1, \dots, n$ in area $a(i) \in \{1, \dots, A\}$:

**Outcomes:**
- Survival: $(T_i,\; \delta_i \in \{0,1\})$
- Treatment time: $(T_i^{\text{ttt}},\; \delta_i^{\text{ttt}} \in \{0,1\})$

**Semi-competing risks:** Death censors treatment. The observed treatment
follow-up time is $\min(T_i^{\text{ttt}},\; T_i,\; C_i)$ where $C_i$ is
administrative censoring time.

**Time-dependent covariates:**

$$Z_i(t) = \mathbf{1}(t \ge T_i^{\text{ttt}}) \quad \text{(treatment indicator)}$$

**Covariate sets** (separate by model sub-process):

| Model | Covariates |
|---|---|
| Survival | Age (per 10yr, centred), CCI, log tumour size, stage II/III dummy |
| Treatment | Age (per 10yr, centred), CCI, SES (area-level), sex, stage II/III dummy |

The covariate sets differ deliberately: survival covariates emphasise clinical
prognosis; treatment covariates emphasise access factors.

### 3.2 Time convention and interval grids

All times are stored in **months** (days ÷ 30.4375). Maximum follow-up is
60 months (5 years).

| Grid | Breakpoints (months) | Purpose |
|---|---|---|
| `SURV_BREAKS` | 0,1,2,3,6,9,12,18,24,36,48,60 | Survival baseline hazard (K=11 intervals) |
| `TTT_BREAKS` | 0,1,2,3,6,9,12,18,24,36,60 | Treatment baseline hazard (K=10 intervals) |
| `POST_TTT_BREAKS` | 0,3,6,12,24,60 | Piecewise post-treatment effect (K=5 intervals) |

### 3.3 Piecewise exponential likelihood

Data are expanded to long format (one row per subject-interval cell). Each
row $r$ has exposure $E_r$ (person-months) and event indicator $y_r \in
\{0,1\}$. The likelihood is Poisson:

$$y_r \sim \text{Poisson}(E_r \cdot \lambda_r)$$

equivalently written as:

$$\ell = \sum_r \left[ y_r \eta_r - E_r e^{\eta_r} \right]$$

where $\eta_r = \log E_r + \log \lambda_r$ is the log-rate for row $r$.

### 3.4 Survival hazard model

$$\lambda_i^{\text{surv}}(t) = \exp\!\Big(
    \alpha_{k(t)}
    + x_i^{\text{surv}\top} \beta
    + \beta_{\text{td}}\, Z_i(t)
    + \delta^{\text{post}}_{k_{\text{post}}(t,i)}\, Z_i(t)
    + u^{\text{surv}}_{a(i)}
\Big)$$

where:
- $\alpha_k = \log \lambda_{0k}^{\text{surv}}$ — log baseline hazard for survival interval $k$
- $\beta$ — fixed effects for clinical covariates
- $\beta_{\text{td}}$ — log hazard ratio for treatment receipt (at treatment onset)
- $\delta^{\text{post}}_k$ — piecewise post-treatment log hazard adjustments; $\delta^{\text{post}}_0 \equiv 0$ (reference: first post-treatment interval); $k \in \{0,\dots,4\}$ indexes `POST_TTT_BREAKS`
- $u^{\text{surv}}_{a(i)}$ — area-level spatial frailty (BYM2)

The piecewise post-treatment term replaces the linear
$\beta_{\text{time}} \cdot (t - T_i^{\text{ttt}})$ term from the original
frequentist model, allowing non-monotone post-treatment hazard dynamics.

### 3.5 Treatment hazard model

$$\lambda_i^{\text{ttt}}(t) = \exp\!\Big(
    \gamma_{k(t)}
    + x_i^{\text{ttt}\top} \theta
    + u^{\text{ttt}}_{a(i)}
\Big)$$

where:
- $\gamma_k$ — log baseline hazard for treatment interval $k$
- $\theta$ — fixed effects for access covariates
- $u^{\text{ttt}}_{a(i)}$ — area-level spatial frailty (BYM2)

### 3.6 BYM2 spatial frailty structure

Each area-level frailty is decomposed into spatially structured and
unstructured components (Riebler et al. 2016):

$$u^{(s)}_a = \sigma_s \!\left(
    \sqrt{\phi_s}\; \tilde{s}^{(s)}_a
    + \sqrt{1 - \phi_s}\; \epsilon^{(s)}_a
\right), \quad s \in \{\text{surv}, \text{ttt}\}$$

where:
- $\tilde{s}^{(s)}$ — scaled ICAR component (see §3.7); $\sum_a \tilde{s}^{(s)}_a = 0$
- $\epsilon^{(s)} \sim \mathcal{N}(0, I_A)$ — unstructured IID component
- $\sigma_s > 0$ — total frailty standard deviation
- $\phi_s \in (0,1)$ — proportion of variance that is spatially structured

$\phi_s \to 1$ means the frailty field is entirely spatial (smooth across
neighbouring counties); $\phi_s \to 0$ means entirely unstructured
(area-specific noise).

### 3.7 ICAR prior and BYM2 scaling

The scaled ICAR component is:

$$\tilde{s}^{(s)} = s^{(s)} / \sqrt{\texttt{scaling\_factor}}$$

where $s^{(s)}$ follows an intrinsic conditional autoregressive (ICAR) prior:

$$p(s) \propto \exp\!\left(-\frac{1}{2} \sum_{(a,b) \in \text{edges}} (s_a - s_b)^2 \right)$$

and `scaling_factor` $= \text{geomean}(\text{diag}(Q^+))$ is the geometric
mean of the diagonal of the Moore-Penrose pseudoinverse of the graph Laplacian
$Q = D - W$. This ensures $\text{Var}(\tilde{s}_a) \approx 1$ marginally,
making $\sigma_s$ interpretable regardless of graph topology.

The sum-to-zero constraint $\sum_a s_a = 0$ is enforced by parameterising with
$A-1$ free components and deriving the last as their negative sum (see
`SpatialGraph.free_to_full`).

### 3.8 Cross-process spatial correlation

The two spatial frailty fields are linked via a Cholesky factor decomposition:

$$u^{\text{ttt}}_a = \rho_u \cdot u^{\text{surv}}_a
    + \sqrt{1 - \rho_u^2} \cdot \tilde{u}^{\text{ttt}}_a$$

where $\tilde{u}^{\text{ttt}}$ is an **independent** BYM2 field with its own
$(\sigma_{\text{ttt}}, \phi_{\text{ttt}})$ parameters.

This parameterisation:
- Makes $\rho_u \in (-1, 1)$ directly interpretable as a correlation
- Keeps the two spatial fields independent in the sampler (avoids the
  near-singular joint precision matrix of the Kronecker form)
- Allows each process to have its own spatial smoothing ($\phi_{\text{surv}}$
  vs $\phi_{\text{ttt}}$)

$\rho_u > 0$ means counties where patients receive timely treatment also tend
to have lower baseline mortality — the *healthy access hypothesis*.

### 3.9 Prior specification

| Parameter | Prior | Justification |
|---|---|---|
| $\alpha_k$ (survival log-baseline hazards) | $\mathcal{N}(-3, 1)$ | $e^{-3} \approx 0.05$/month; weakly informative on month scale |
| $\gamma_k$ (treatment log-baseline hazards) | $\mathcal{N}(-3, 1)$ | Same reasoning |
| $\beta$ (survival fixed effects) | $\mathcal{N}(0, 1)$ | Log-HR scale; ±2 covers plausible clinical effect sizes |
| $\theta$ (treatment fixed effects) | $\mathcal{N}(0, 1)$ | Same |
| $\beta_{\text{td}}$ (treatment indicator) | $\mathcal{N}(0, 1)$ | Log-HR; true value ≈ −0.25 in simulation |
| $\delta_k^{\text{post}}$ (piecewise post-ttt) | $\mathcal{N}(0, 0.5)$ | Tighter: deviations from $\beta_{\text{td}}$ should be modest |
| $\sigma_{\text{surv}}$ | $\text{HalfNormal}(0, 1)$ | True $\sigma=1$ in simulation |
| $\sigma_{\text{ttt}}$ | $\text{HalfNormal}(0, 0.5)$ | True $\sigma=0.3$ in simulation |
| $\phi_{\text{surv}}, \phi_{\text{ttt}}$ | $\text{Beta}(0.5, 0.5)$ | U-shaped; lets data determine spatial fraction |
| $\rho_u$ | $\text{Uniform}(-1, 1)$ | Flat on correlation scale; identification from data |

### 3.10 Full model parameter inventory

| Block | Parameters | Count |
|---|---|---|
| Survival log-baseline hazards | $\alpha_0, \dots, \alpha_{10}$ | 11 |
| Treatment log-baseline hazards | $\gamma_0, \dots, \gamma_9$ | 10 |
| Survival fixed effects | $\beta$ | 4 |
| Treatment fixed effects | $\theta$ | 5 |
| Treatment indicator | $\beta_{\text{td}}$ | 1 |
| Post-treatment piecewise (excl. ref.) | $\delta^{\text{post}}_1, \dots, \delta^{\text{post}}_4$ | 4 |
| Spatial variance and mixing | $\sigma_{\text{surv}}, \phi_{\text{surv}}, \sigma_{\text{ttt}}, \phi_{\text{ttt}}$ | 4 |
| Cross-process correlation | $\rho_u$ | 1 |
| Survival ICAR free params | $\tilde{s}^{\text{surv}}_1, \dots, \tilde{s}^{\text{surv}}_{A-1}$ | $A-1$ |
| Survival IID params | $\epsilon^{\text{surv}}_1, \dots, \epsilon^{\text{surv}}_A$ | $A$ |
| Treatment ICAR free params | $\tilde{s}^{\text{ttt}}_1, \dots, \tilde{s}^{\text{ttt}}_{A-1}$ | $A-1$ |
| Treatment IID params | $\epsilon^{\text{ttt}}_1, \dots, \epsilon^{\text{ttt}}_A$ | $A$ |
| **Total (A=159)** | | **≈ 677** |

All frailty parameters use **non-centered parameterisation** throughout
(ICAR and IID components sampled as standard normals, then scaled).

---

## 4. Package architecture

```
SEER_PEPH/
├── pyproject.toml               Package metadata and dependencies
├── README.md                    Quick-start and overview
├── DEVELOPER_GUIDE.md           This document
│
├── src/seer_peph/
│   ├── __init__.py              Version exposure
│   │
│   ├── graphs.py                ✅ Complete
│   │   Adjacency graph utilities for BYM2 spatial frailty:
│   │   SpatialGraph dataclass, make_ring_lattice(), make_grid(),
│   │   from_adjacency(), from_edge_csv(), load_georgia_counties()
│   │
│   ├── data/
│   │   ├── __init__.py          Re-exports public API
│   │   ├── prep.py              ✅ Complete
│   │   │   Wide → long-format Poisson expansion for both sub-models.
│   │   │   load_and_encode(), build_survival_long(),
│   │   │   build_treatment_long(), build_area_map(), main()
│   │   └── model_data.py        🔲 Planned (Phase 1)
│   │       DataFrame → JAX array contract for NumPyro models.
│   │
│   ├── simulate.py              🔲 Planned (before Phase 1)
│   │   Data-generating process for all simulation phases.
│   │   Parameterised by (A, n_per_area, rho_u, phi_surv, phi_ttt, ...)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── priors.py            🔲 Planned (Phase 1)
│   │   │   Shared prior specifications as NumPyro distributions.
│   │   ├── survival_only.py     🔲 Planned (Phase 1)
│   │   │   PE survival model, fixed effects + piecewise post-ttt,
│   │   │   no frailties. Competitor Model M1.
│   │   ├── joint_iid.py         🔲 Planned (Phase 2)
│   │   │   Joint model with IID correlated frailties, no spatial
│   │   │   structure. Competitor Model M2.
│   │   ├── joint_bym2.py        🔲 Planned (Phase 3+4)
│   │   │   Phase 3: independent BYM2 frailties (rho_u=0). M3.
│   │   │   Phase 4: full joint BYM2 with rho_u. Primary model M4.
│   │   └── base.py              🔲 Planned (Phase 1)
│   │       Shared log-likelihood helpers (PE Poisson log-lik,
│   │       ICAR factor, BYM2 field constructor).
│   │
│   └── inference/
│       ├── __init__.py
│       ├── run.py               🔲 Planned (Phase 1)
│       │   NUTS runner: warmup/sample config, diagnostics (R-hat,
│       │   ESS, divergences), posterior serialisation.
│       └── sbc.py               🔲 Planned (Phase 5)
│           Simulation-based calibration: rank computation,
│           uniformity tests, rank histogram plotting.
│
└── tests/
    ├── data/
    │   └── test_prep.py         ✅ Complete (37 unit tests)
    ├── test_graphs.py           🔲 Planned
    ├── test_simulate.py         🔲 Planned
    └── models/
        └── test_likelihoods.py  🔲 Planned
```

### Dependency separation

The package is split into two dependency tiers:

```
Core (always installed)          Inference (optional: pip install -e ".[inference]")
─────────────────────────        ──────────────────────────────────────────────────
numpy, pandas, scipy             numpyro, jax, jaxlib
  data/prep.py                     models/*.py
  graphs.py                        inference/*.py
  simulate.py (DGP only)           simulate.py (full use)
```

`data/prep.py` and `graphs.py` have no JAX dependency. This means data
preparation and graph construction can be run, tested, and profiled without
the inference stack installed.

---

## 5. Module reference

### `seer_peph.graphs`

Adjacency graph representation and BYM2 scaling.

```python
from seer_peph.graphs import make_ring_lattice, make_grid, from_adjacency

# Synthetic graphs for simulation
g20  = make_ring_lattice(20,  k=4)   # Phase 1-5 development
g50  = make_ring_lattice(50,  k=4)   # Phase 6 scaling study
g159 = make_ring_lattice(159, k=4)   # Phase 6 GA proxy

# Key attributes passed to NumPyro model as static data
g20.node1           # int32[n_edges]  — edge list
g20.node2           # int32[n_edges]
g20.scaling_factor  # float           — BYM2 scale
g20.A               # int             — number of areas

# Inside a NumPyro model
s_free = numpyro.sample("s_free", dist.Normal(0, 1).expand([graph.A - 1]))
s_full = graph.free_to_full(s_free)   # sum-to-zero constraint
numpyro.factor("icar", -0.5 * jnp.sum((s_full[graph.node1] - s_full[graph.node2])**2))
```

### `seer_peph.data`

Data loading, encoding, and long-format expansion.

```python
from seer_peph.data import main, load_and_encode, build_survival_long

# Full pipeline
data = main("path/to/data.csv")
surv_long = data["surv_long"]   # survival Poisson rows
ttt_long  = data["ttt_long"]    # treatment Poisson rows

# Key columns in surv_long
# id, k, t0, t1, exposure, event     — standard PE columns
# treated_td                          — time-dependent treatment indicator
# k_post                              — post-ttt interval index (-1 if untreated)
# area_id                             — 0-based area index
# age_per10_centered, cci, tumor_size_log, stage_II, stage_III

# Key columns in ttt_long
# id, k, t0, t1, exposure, event
# area_id
# age_per10_centered, cci, ses, sex_male, stage_II, stage_III
```

### `seer_peph.data.model_data` *(planned)*

Converts `prep.py` DataFrames to JAX arrays. The contract is:

```python
from seer_peph.data.model_data import make_model_data

model_data = make_model_data(surv_long, ttt_long, graph)
# Returns dict of JAX arrays:
# Survival arrays:
#   y_surv, log_exposure_surv, k_surv, k_post, treated_td,
#   area_id_surv, X_surv
# Treatment arrays:
#   y_ttt, log_exposure_ttt, k_ttt, area_id_ttt, X_ttt
# Graph arrays (constants):
#   node1, node2, scaling_factor, A
```

### `seer_peph.simulate` *(planned)*

Data-generating process for all simulation phases. Core signature:

```python
from seer_peph.simulate import simulate_joint

data = simulate_joint(
    graph,                    # SpatialGraph
    n_per_area    = 500,
    rho_u         = 0.85,
    phi_surv      = 0.80,
    phi_ttt       = 0.75,
    sigma_surv    = 1.0,
    sigma_ttt     = 0.3,
    beta_td       = -0.25,    # treatment effect
    seed          = 42,
)
# Returns wide-format DataFrame with same structure as the
# simulated SEER dataset, including *_true columns for validation.
```

---

## 6. Implementation roadmap

| Phase | Goal | Status | Competitor model |
|---|---|---|---|
| 0 | Data pipeline (`prep.py`) | ✅ Complete | — |
| 0b | Graph utilities (`graphs.py`) | ✅ Complete | — |
| 0c | DGP (`simulate.py`) | 🔲 Next | — |
| 0d | Array contract (`model_data.py`) | 🔲 Next | — |
| 1 | Survival-only, no frailties | 🔲 | M1 |
| 2 | Joint model, IID correlated frailties | 🔲 | M2 |
| 3 | Independent BYM2 frailties (ρ_u = 0) | 🔲 | M3 |
| 4 | Full joint BYM2 model | 🔲 | M4 (primary) |
| 5 | Simulation-based calibration (SBC) | 🔲 | — |
| 6 | Methods paper simulation study | 🔲 | All four |
| 7 | Real Georgia SEER-Medicare analysis | 🔲 | — |

### Phase exit criteria

Each phase must pass before the next begins:

| Phase | Exit criterion |
|---|---|
| 1 | β_td 90% CI covers true −0.25; all δ^post_k recovered; R-hat < 1.01 |
| 2 | ρ_u 90% CI covers true 0.85; no divergences; ESS > 400 for variance params |
| 3 | φ posteriors concentrated above 0.5; area frailty recovery correlation > 0.7 |
| 4 | All phase 2+3 criteria; no pathological geometry in (ρ_u, φ, σ) pairs plots |
| 5 | Rank histograms uniform for ρ_u, β_td, σ_surv, σ_ttt, φ_surv, φ_ttt |
| 6 | M2 shows biased ρ_u under S2 (strong spatial + true ρ_u=0.85) |

---

## 7. Simulation study design

The simulation study has two distinct roles:

### 7.1 Model validation (Phase 5 — SBC)

Simulation-based calibration confirms the Bayesian implementation is correct:
posteriors must be calibrated, meaning the 90% credible interval covers the
true value 90% of the time across repeated draws from the prior-predictive
distribution.

**Protocol:**
1. Draw parameters from the prior
2. Simulate data via `simulate_joint()`
3. Fit the full model (Phase 4)
4. Compute rank of true parameter within posterior draws
5. Repeat S = 500 times
6. Test rank uniformity with χ² test; plot rank histograms

**Parameters requiring SBC:**
ρ_u, β_td, δ^post_1..4, σ_surv, σ_ttt, φ_surv, φ_ttt, representative u^surv_a

### 7.2 Methods paper contribution (Phase 6)

**The core result:** Under strong spatial structure (high φ, scenario S2),
the joint IID model (M2) produces biased and overconfident estimates of ρ_u.
The full joint BYM2 model (M4) recovers ρ_u correctly.

**Scenarios:**

| Scenario | Spatial structure | ρ_u | Description |
|---|---|---|---|
| S1 | None (IID) | 0.85 | No spatial structure — IID frailties only |
| S2 | BYM2, φ = 0.85 | 0.85 | Strong spatial structure |
| S3 | BYM2, φ = 0.30 | 0.85 | Weak spatial structure |
| S4 | BYM2, φ = 0.85 | 0.00 | Null case — no cross-process correlation |

**Models fitted to each scenario:**

| Model | Description |
|---|---|
| M1 | No frailties (survival only) |
| M2 | Joint IID correlated frailties, no spatial structure |
| M3 | Independent BYM2 frailties, ρ_u fixed at 0 |
| M4 | Full joint BYM2 model with ρ_u (primary model) |

**Estimands tracked (200 replications per cell):**
- Posterior median and 90% CI width for ρ_u
- Coverage of 90% CI for ρ_u
- Posterior median bias for β_td

**Scaling study parameter grid:**

| Factor | Levels |
|---|---|
| A (areas) | 20, 50, 159 |
| n/A (patients per area) | 50, 250, 500 |
| φ (spatial mixing) | 0.2, 0.5, 0.85 |
| ρ_u | 0, 0.5, 0.85 |

Full factorial = 54 cells × 200 reps = 10,800 fits. Use ADVI/Pathfinder for
the scaling study; confirm key cells with NUTS.

### 7.3 Identifiability considerations

The effective sample size for estimating ρ_u is approximately the number of
areas with sufficient within-area events to produce informative frailty
estimates (rough threshold: ≥ 30–50 events per area). For real Georgia
SEER-Medicare data this is approximately 40–70 of the 159 counties, due to
sparse rural counties.

The scientific claim requires $P(\rho_u > 0 \mid \text{data})$ to be high
(e.g. > 0.95), not a precise point estimate. A wide posterior concentrated
above zero is sufficient.

**Bayesian power study** (run before real data analysis):
Simulate with realistic Georgia county-size distribution (log-normal, mean ~90
patients/county, heavy left tail). Report $P(\rho_u > 0)$ under true
ρ_u ∈ {0, 0.5, 0.85} as a function of study period length.

---

## 8. Development setup

### Installation

```bash
git clone <repo>
cd SEER_PEPH

# Data pipeline only (no JAX)
pip install -e ".[dev]"

# Full inference stack
pip install -e ".[inference,dev]"
```

### Running tests

```bash
# All tests (requires test data at default path or env var)
pytest

# With simulated dataset
SEER_PEPH_TEST_DATA=path/to/joint_ttt_survival_dataset.csv pytest

# Specific module
pytest tests/data/test_prep.py -v
```

### Environment variable

`SEER_PEPH_TEST_DATA` — path to the simulated SEER-like CSV. Tests in
`TestPipeline` are skipped automatically when the file is not present.

### Typical development workflow

```python
import sys
sys.path.insert(0, "src")

from seer_peph.data import main
from seer_peph.graphs import make_ring_lattice

# Load and expand data
data = main("joint_ttt_survival_dataset.csv")
surv_long = data["surv_long"]
ttt_long  = data["ttt_long"]

# Build simulation graph
graph = make_ring_lattice(20, k=4)
print(graph.summary())

# (Phase 1+) Build model data arrays
from seer_peph.data.model_data import make_model_data
model_data = make_model_data(surv_long, ttt_long, graph)
```

---

## 9. Conventions and standards

### Time unit
All times are in **months** throughout the codebase. Raw SEER data in days is
converted in `load_and_encode()` via `DAYS_PER_MONTH = 365.25 / 12`. Never
mix days and months within a module.

### Area indexing
Area identifiers (`area_id`) are always **0-based contiguous integers**
matching the row/column indices of `SpatialGraph.adjacency`. The mapping from
zip code or FIPS code to `area_id` is maintained in `build_area_map()`.

### Parameterisation
- All frailty components use **non-centered parameterisation** — raw standard
  normals are sampled and scaled inside the model, never the frailties directly.
- ICAR fields are **always sum-to-zero** — use `graph.free_to_full()`.
- Log-scale parameters (`alpha`, `gamma`) are parameterised directly as
  log-hazards, not as hazards with a positivity constraint.

### JAX / NumPy interoperability
- Modules without JAX dependency (`graphs.py`, `data/prep.py`) must work with
  NumPy arrays only.
- Modules with JAX dependency (`models/`, `inference/`) should accept JAX
  arrays and use `jnp` throughout.
- `SpatialGraph.free_to_full()` and `SpatialGraph.icar_logdens()` dispatch
  automatically based on input array type.

### Column contract
The long-format DataFrames from `prep.py` are the single source of truth for
column names. `model_data.py` reads exactly these columns and no others:

```
surv_long required : id, k, exposure, event, treated_td, k_post,
                     area_id, age_per10_centered, cci, tumor_size_log,
                     stage_II, stage_III

ttt_long required  : id, k, exposure, event,
                     area_id, age_per10_centered, cci, ses,
                     sex_male, stage_II, stage_III
```

### Adding a new model
1. Create `src/seer_peph/models/your_model.py`
2. Define a single `model(data)` function that takes the `model_data` dict
   and contains the full NumPyro probabilistic program
3. Add a corresponding runner call in `inference/run.py`
4. Add at least one test in `tests/models/` that fits the model on a tiny
   synthetic dataset (A=5, n=50) and checks that it samples without error

### Git conventions
- Branch naming: `phase-N-description` (e.g. `phase-1-survival-model`)
- Commit messages: imperative mood, reference phase
  (e.g. `Add PE log-likelihood helper [Phase 1]`)
- Every PR must pass all existing tests before merge

---

## 10. Key references

**BYM2 spatial prior:**  
Riebler A, Sørbye SH, Simpson D, Rue H (2016). An intuitive Bayesian spatial
model for disease mapping that accounts for scaling. *Statistical Methods in
Medical Research*, 25(4), 1145–1165.

**ICAR prior and spatial survival models:**  
Banerjee S, Wall MM, Carlin BP (2003). Frailty modeling for spatially
correlated survival data, with application to infant mortality in Minnesota.
*Biostatistics*, 4(1), 123–142.

**Piecewise exponential survival:**  
Friedman M (1982). Piecewise exponential models for survival data with
covariates. *The Annals of Statistics*, 10(1), 101–113.

**Joint models for semi-competing risks:**  
Fine JP, Jiang H, Chappell R (2001). On semi-competing risks data.
*Biometrika*, 88(4), 907–919.

**Simulation-based calibration:**  
Talts S, Betancourt M, Simpson D, Vehtari A, Gelman A (2020). Validating
Bayesian inference algorithms with simulation-based calibration.
*arXiv:1804.06788*.

**NumPyro:**  
Phan D, Pradhan N, Jankowiak M (2019). Composable effects for flexible and
accelerated probabilistic programming in NumPyro. *arXiv:1912.11554*.

**Bayesian spatial joint frailty (background):**  
Duchateau L, Janssen P (2008). *The Frailty Model*. Springer.
