# SEER-PEPH Developer Guide

Updated for the current implementation state of the SEER-PEPH joint treatment-survival modeling project.

## 1. Purpose and scope

SEER-PEPH is a research package for developing, validating, and ultimately applying Bayesian joint treatment-survival models to SEER-Medicare colorectal cancer data with spatial structure. The package is designed around four principles:

1. **Explicit preprocessing contracts.** Wide subject-level data are transformed into stable long-format Poisson representations for both survival and treatment-time processes.
2. **Simulation-driven model development.** New model components are introduced only after they are validated against known truth on simulated data.
3. **Thin, reusable interfaces.** The same array contract is shared across standalone survival models, standalone treatment models, and the eventual joint model.
4. **A path to causal-style treatment-timing questions.** The end goal is not only estimation, but also prediction under alternative treatment-timing scenarios.

This guide is written to help a newcomer understand how the current code base works, what is already stable, what remains under active development, and how to add new components without breaking the existing workflow.

---

## 2. Current project status

At the current stage, the project has the following implemented or effectively stabilized components:

- preprocessing of wide data into survival and treatment long formats
- spatial graph construction and validation
- simulation of joint treatment-survival data with correlated spatial frailties
- model-data assembly into a stable NumPyro/JAX array contract
- a validated standalone **spatial delta-only survival model**
- a validated standalone **spatial treatment-time piecewise-exponential model**
- single-seed and multi-seed recovery workflows
- null and diagnostic scripts for the survival component
- interval-support diagnostics for the treatment component

The current leading survival model is:

- piecewise-exponential survival likelihood
- baseline covariate effects
- **delta-only** post-treatment effect parameterization
- BYM2-style spatial frailty
- independent priors on `delta_post`
- **no separate `beta_td` term**

The current standalone treatment model is:

- piecewise-exponential treatment-time likelihood
- baseline treatment covariate effects
- BYM2-style spatial frailty
- independent baseline interval effects `gamma`

The next major development target is the **full joint treatment-survival model**, implemented by coupling the two validated standalone submodels through correlated spatial frailties.

---

## 3. High-level package architecture

The project is organized around a small number of modules with clear responsibilities.

### 3.1 `seer_peph.data.prep`

This module performs the data-engineering work that converts subject-level wide data into long-format Poisson representations.

Its responsibilities include:

- defining the default time grids used by the package
- loading and encoding wide data
- constructing `area_id`
- building long-format survival data
- building long-format treatment-time data
- enforcing row semantics around exposures, event assignment, treatment status, and post-treatment interval indexing

The module defines the core time grids:

- `SURV_BREAKS`: survival piecewise-exponential intervals
- `TTT_BREAKS`: treatment-time piecewise-exponential intervals
- `POST_TTT_BREAKS`: intervals for time since treatment in the survival model

It also defines default covariate column lists:

- `SURV_X_COLS`
- `TTT_X_COLS`

The long-format outputs are:

- `surv_long`: one row per subject-interval cell for the survival model
- `ttt_long`: one row per subject-interval cell for the treatment-time model

This module is the most important place to understand if you need to change row semantics, treatment-history coding, or interval design.

### 3.2 `seer_peph.data.model_data`

This module converts the pandas long-format tables into the array contract consumed by NumPyro model files.

Its responsibilities include:

- validating long-format inputs
- selecting and ordering design-matrix columns
- building survival arrays
- building treatment arrays
- attaching graph constants
- returning a single dictionary with stable keys and metadata

The point of this module is to isolate all data-shape and dtype concerns from the model definitions.

### 3.3 `seer_peph.graphs`

This module defines spatial graph structures and utilities.

Its responsibilities include:

- creating synthetic development graphs, especially ring lattices
- validating adjacency/indexing structure
- providing the graph scaling factor used in ICAR/BYM2-like priors

For simulation and validation, synthetic graphs are currently the primary setting.

### 3.4 `seer_peph.validation.simulate`

This module generates simulated joint treatment-survival data.

Its responsibilities include:

- generating baseline covariates
- generating correlated spatial frailties for treatment and survival
- simulating treatment timing
- simulating survival with treatment-history effects
- exporting truth columns used in recovery studies

This module is central to the project. Nearly every major model-development step should be checked against it.

### 3.5 `seer_peph.models`

This package contains the NumPyro model definitions.

The important current and near-current files are:

- `survival_spatial_delta_only.py`: current leading standalone survival model
- `treatment_spatial_pe.py`: standalone treatment-time model used for treatment recovery work
- `joint_spatial_treatment_survival.py`: first full joint model target, coupling the validated submodels through correlated spatial frailties

The survival and treatment standalone models are already aligned with the shared data contract and should be treated as the two stable building blocks for the joint model.

### 3.6 `seer_peph.inference.run`

This module provides a thin wrapper around NumPyro MCMC.

Its responsibilities include:

- storing inference settings in a typed config object
- building and running NUTS
- returning posterior samples and diagnostics
- summarizing posterior samples into scalar summaries for downstream scripts

This wrapper intentionally does not hide NumPyro. It standardizes the workflow while keeping the model files transparent.

### 3.7 `scripts/`

The scripts directory is the executable validation and diagnostics layer of the project.

It includes workflows for:

- single-seed recovery
- multi-seed recovery
- null recovery
- targeted diagnostics for weak parameter recovery
- interval-support summaries

These scripts should be treated as part of the package-level validation suite, not as disposable experiments.

---

## 4. End-to-end data flow

The package currently follows this conceptual pipeline:

1. **Wide data** at the subject level are loaded or simulated.
2. The wide data are encoded into analysis-ready columns such as:
   - months-scale times
   - `sex_male`
   - `stage_II`
   - `stage_III`
   - `area_id`
3. The wide data are expanded into:
   - `surv_long`
   - `ttt_long`
4. `make_model_data(...)` converts these long tables plus a graph into a dictionary of arrays.
5. A NumPyro model consumes the dictionary and produces posterior samples.
6. Recovery and diagnostic scripts compare posterior summaries to known truth.

The practical consequence is that model files do **not** work directly with raw pandas wide data. They consume only the long-format array contract produced by `make_model_data(...)`.

---

## 5. Core time grids and row semantics

The package depends heavily on the distinction between three time grids.

### 5.1 Survival baseline grid

`SURV_BREAKS` defines the baseline hazard intervals for the survival process.

These are the intervals indexed by `k_surv` in the survival long data and by `alpha[k]` in the current survival model.

### 5.2 Treatment-time grid

`TTT_BREAKS` defines the baseline hazard intervals for the treatment-time process.

These are the intervals indexed by `k_ttt` in the treatment long data and by `gamma[k]` in the standalone treatment model.

### 5.3 Post-treatment grid

`POST_TTT_BREAKS` defines intervals in **time since treatment initiation** used by the survival model.

These are not diagnosis-time intervals. They are intervals on the post-treatment timescale. They determine `k_post`, which is used to index `delta_post`.

This distinction is fundamental. A survival row can be in diagnosis-time interval `k_surv = 6` and simultaneously in post-treatment interval `k_post = 1`, depending on when treatment started.

---

## 6. The preprocessing layer in detail

### 6.1 What `prep.py` does

`prep.py` is where the continuous-time clinical processes are converted into a piecewise-exponential representation.

The public API includes:

- `load_and_encode(path)`
- `build_survival_long(df)`
- `build_treatment_long(df)`
- `build_area_map(df)`
- `main(path)`

### 6.2 Survival long-format semantics

For each survival long row, the important fields are:

- `id`: subject identifier
- `k`: diagnosis-time survival interval index
- `exposure`: time at risk within that interval
- `event`: event indicator for that row
- `treated_td`: whether the row is after treatment initiation
- `k_post`: post-treatment interval index, or `-1` if pre-treatment
- covariate columns copied from the subject-level record

The survival long builder inserts cut points at:

- the global survival breaks
- the treatment time, if present
- the relevant post-treatment break boundaries

This guarantees that a survival long row lies wholly inside one baseline survival interval, one treatment-status state, and one post-treatment interval.

### 6.3 Treatment long-format semantics

For each treatment long row, the important fields are:

- `id`
- `k`: treatment baseline interval index
- `exposure`
- `event`
- covariate columns copied from the subject-level record

The treatment process is built under semi-competing-risk semantics:

- treatment is observed only while the subject remains alive and uncensored
- death censors treatment
- treatment follow-up is truncated at `min(treatment_time, death_time, administrative censoring)`

This is why treatment interval support thins out in later intervals: fewer subjects remain in the treatment risk set long enough to contribute person-time and events.

### 6.4 Important helper logic in `prep.py`

The main helper functions worth understanding are:

- `_interval_index(...)`: maps a time to a PE interval
- `_event_interval_index(...)`: assigns an event to the interval that closes at the event time
- `_merged_breaks(...)`: merges global breakpoints with subject-specific cut times
- `_post_ttt_k(...)`: computes the post-treatment interval index at the start of a survival long row
- `_expand(...)`: core engine used by both long-format builders

If a contributor needs to modify interval semantics, event assignment, or post-treatment row coding, this is where the work belongs.

---

## 7. The model-data contract

`model_data.py` defines the stable interface between pandas preprocessing and JAX/NumPyro model code.

### 7.1 Returned dictionary keys

The returned dictionary contains the following blocks.

#### Survival block

- `y_surv`: event counts
- `log_exposure_surv`
- `k_surv`
- `k_post`
- `treated_td`
- `area_id_surv`
- `X_surv`

#### Treatment block

- `y_ttt`
- `log_exposure_ttt`
- `k_ttt`
- `area_id_ttt`
- `X_ttt`

#### Graph block

- `node1`
- `node2`
- `scaling_factor`
- `A`

#### Metadata block

- `surv_x_cols`
- `ttt_x_cols`
- `N_surv`, `N_ttt`
- `P_surv`, `P_ttt`

### 7.2 Design philosophy

This module validates aggressively. It is intended to catch shape mismatches and inconsistent indexing before they reach NumPyro. That means model files can safely assume:

- integer indices are in range
- exposures are valid and finite
- design matrices have the expected shape
- graph constants are aligned with area indices

### 7.3 Why this matters

The main long-term advantage of this design is that every model file can rely on the same upstream contract. That makes it easier to compare survival variants, treatment variants, and joint variants without rewriting preprocessing logic.

---

## 8. Spatial graph layer

The graph utilities provide the spatial structure required by the BYM2-style priors.

The most important object is the graph itself, which provides:

- number of areas `A`
- edge list structure
- a scaling factor used to stabilize the ICAR-like structured component

The current validation work has primarily used ring-lattice graphs via `make_ring_lattice(...)`, because they provide a controlled synthetic setting with nontrivial adjacency structure.

Contributors adding real spatial graphs should preserve the same interface expected by `make_model_data(...)`.

---

## 9. Simulation framework

The simulator is the foundation of the validation strategy.

### 9.1 What it simulates

The simulation module generates:

- baseline covariates
- a survival spatial frailty field
- a treatment spatial frailty field correlated with the survival field
- treatment times from a piecewise-exponential treatment hazard
- survival times from a piecewise-exponential survival hazard with post-treatment effects

### 9.2 Main parameter blocks

The simulator uses truth settings for:

- `beta_surv`: baseline survival covariate effects
- `gamma_ttt`: treatment baseline interval effects
- `theta_ttt`: treatment covariate effects
- `delta_post`: full post-treatment survival effects by post-treatment interval
- `phi_surv`, `sigma_surv`: survival BYM2 parameters
- `phi_ttt`, `sigma_ttt`: treatment BYM2 parameters
- `rho_u`: cross-process frailty correlation parameter

### 9.3 Important simulator conventions

The simulator currently follows the convention that the survival treatment effect is represented operationally as:

- a full post-treatment effect per interval
- consistent with the current **delta-only** fitted survival parameterization

It also generates the treatment frailty by coupling the survival frailty and an independent treatment frailty field. This is the template for the first full joint model.

### 9.4 Exported truth columns

The wide simulated data include columns such as:

- `u_surv_true`
- `u_ttt_true`
- `phi_surv_true`, `phi_ttt_true`
- `sigma_surv_true`, `sigma_ttt_true`
- `theta_ttt_*_true`
- `delta_post_j_true`
- `gamma_ttt_j_true` for treatment baseline intervals

These truth columns are used by the recovery scripts.

---

## 10. Current leading survival model

The current active survival model is `survival_spatial_delta_only.py`.

### 10.1 Mathematical form

For survival long row `r`:

\[
 y_r \sim \text{Poisson}(e_r \lambda_r)
\]

\[
\log(\lambda_r) = \alpha_{k_r} + x_r^\top \beta + u_{a_r} + \text{treated}_r\,\delta_{k_{post,r}}
\]

where:

- `alpha[k]` is the baseline survival log-hazard
- `beta` are baseline survival covariate effects
- `u[a]` is the survival spatial frailty
- `delta_post[j]` is the **full** post-treatment log-hazard shift in post-treatment interval `j`

### 10.2 Why the delta-only form is preferred

Earlier work explored a parameterization with a separate acute effect term plus interval-specific deviations. That behaved worse in recovery studies. The current delta-only parameterization is preferred because it is simpler, more stable, and better aligned with the scientific target of comparing treatment-timing trajectories.

### 10.3 Spatial prior structure

The model uses a BYM2-style construction combining:

- an unstructured Gaussian component
- a structured ICAR-like component scaled by the graph factor

The resulting latent area effect is the quantity of primary practical interest. In validation, exact recovery of the mixing and scale hyperparameters is usually less important than recovery of the effective frailty field `u`.

### 10.4 Current status of the survival component

This component should be treated as the currently stabilized survival backbone of the project.

---

## 11. Standalone treatment-time model

The standalone treatment-time model is the treatment-side analogue of the survival model and is now the working treatment module.

### 11.1 Mathematical form

For treatment long row `q`:

\[
 y^{ttt}_q \sim \text{Poisson}(e_q \lambda^{ttt}_q)
\]

\[
\log(\lambda^{ttt}_q) = \gamma_{k_q} + z_q^\top \theta + u^{ttt}_{a_q}
\]

where:

- `gamma[k]` are treatment baseline log-hazards by diagnosis-time interval
- `theta` are treatment covariate effects
- `u_ttt[a]` is the treatment spatial frailty

### 11.2 Design choices

The first standalone treatment model intentionally mirrors the structure of the survival model where possible:

- same array contract
- same general BYM2 frailty design
- same inference wrapper
- validation through simulation and multiseed recovery

The treatment model currently uses independent priors on the baseline `gamma[k]` intervals rather than smoothed priors.

### 11.3 Validation findings

Current treatment validation indicates:

- fixed-effect recovery is broadly good
- treatment spatial frailty recovery is strong
- late baseline intervals `gamma[7:9]` are weaker, largely because interval support thins out in the tail
- exact recovery of spatial hyperparameters is weaker than recovery of the latent frailty field itself

This is acceptable for the current stage. The treatment module is ready to serve as a building block for joint-model development.

---

## 12. Inference layer

The `seer_peph.inference.run` module standardizes fitting.

### 12.1 Main components

The key pieces are:

- `InferenceConfig`
- `run_mcmc(...)`
- `summarise_samples(...)`

`InferenceConfig` stores settings such as:

- number of chains
- warmup iterations
- posterior samples
- target acceptance probability
- mass-matrix choice
- tree depth

### 12.2 Philosophy

The wrapper is intentionally thin. Model definitions remain plain NumPyro model functions that consume a data dictionary. The inference wrapper handles sampler setup and sample summarization without obscuring the statistical model.

This design makes it easy to swap model files while keeping the fitting pattern constant.

---

## 13. Validation scripts and diagnostic workflow

The project relies heavily on script-level validation.

### 13.1 Survival scripts

Important survival-side scripts currently include:

- `run_survival_spatial_delta_only_multi_seed_recovery.py`
- `run_survival_spatial_delta_only_recovery_null.py`
- `run_survival_spatial_delta_only_recovery_null_multi_seed.py`
- `run_survival_spatial_tumor_size_diagnostics.py`
- `run_survival_spatial_delta_only_multi_seed_tumor_diagnostics.py`

These support:

- parameter recovery
- null recovery
- multiseed stability assessment
- targeted evaluation of attenuation in `tumor_size_log`

### 13.2 Treatment scripts

Important treatment-side scripts currently include:

- `run_treatment_spatial_pe_recovery.py`
- a multiseed treatment recovery script built on the same structure
- treatment interval support summaries

These support:

- single-seed treatment recovery
- multiseed parameter recovery
- spatial frailty recovery
- interval-support diagnostics by treatment interval

### 13.3 What the diagnostics have established

The current validation evidence supports the following practical conclusions:

- the delta-only survival model is the preferred survival formulation
- the standalone treatment PE model is adequate as the treatment building block
- the latent spatial frailty fields recover more reliably than the BYM2 hyperparameters themselves
- weak recovery of late treatment baseline intervals is largely attributable to weaker tail support

---

## 14. Recommended interpretation of the current validation state

A contributor joining the project should understand the current status as follows.

### 14.1 What is effectively frozen

The following choices should be treated as the current defaults unless there is strong new evidence to revisit them:

- the survival model should remain the **plain delta-only independent-prior version**
- the older `beta_td + delta_post` survival parameterization should not be revived as the leading model
- the standalone treatment model should remain a plain spatial piecewise-exponential model with independent `gamma[k]`
- the shared `make_model_data(...)` contract should be preserved

### 14.2 What remains open but not urgent

- whether the treatment baseline eventually needs smoothing in the tail
- whether the spatial hyperparameters should be regularized more strongly
- whether subject-level or nonspatial shared frailties will be necessary in later joint-model extensions

### 14.3 What remains actively under development

- the first full joint treatment-survival model
- posterior utilities for counterfactual survival under alternative treatment times
- recovery scripts for the joint model

---

## 15. The first full joint model: current trajectory

The next active model-development stage is the full joint model.

### 15.1 Intended structure

The first joint model should combine the two validated standalone submodels without introducing unnecessary extra dependence structure.

The intended form is:

#### Survival component

\[
\log \lambda^{surv}_r = \alpha_{k_r} + x_r^\top \beta + u^{surv}_{a_r} + \text{treated}_r\,\delta_{k_{post,r}}
\]

#### Treatment component

\[
\log \lambda^{ttt}_q = \gamma_{k_q} + z_q^\top \theta + u^{ttt}_{a_q}
\]

#### Cross-process coupling

\[
 u^{ttt}_a = \rho_u\,u^{surv}_a + \sqrt{1-\rho_u^2}\,u^{ttt,ind}_a
\]

This matches the simulator’s cross-process frailty construction and is therefore the correct first joint target.

### 15.2 Why this is the right next step

This approach:

- reuses already validated survival and treatment components
- preserves the existing preprocessing and model-data contract
- aligns exactly with the simulation design used for recovery
- avoids premature complexity in the dependence structure

### 15.3 What the first joint model should not do

The first joint model should **not** yet introduce:

- a new survival treatment-history parameterization
- smoothed baseline priors by default
- subject-level shared frailties
- additional direct dependence of survival on latent treatment hazard beyond observed treatment history
- complicated shared-field decompositions not already reflected in the simulator

---

## 16. File-by-file guide for contributors

This section summarizes what a contributor is likely to touch and what kinds of changes belong in each file.

### `prep.py`

Touch this file if you need to change:

- time grids
- long-format row semantics
- treatment-history coding
- event assignment logic
- covariates copied into long rows

Do **not** use this file to change statistical priors or model parameterizations.

### `model_data.py`

Touch this file if you need to change:

- the arrays passed into models
- covariate selection for design matrices
- metadata keys used by models and scripts
- validation logic for array contracts

This file is the place to add new model inputs in a way that keeps the interface explicit and stable.

### `graphs.py`

Touch this file if you need to:

- add a new graph-construction utility
- adapt graph validation
- support real spatial adjacency structures

### `simulate.py`

Touch this file if you need to:

- change simulation truth parameters
- add truth exports for new parameters
- modify the dependence structure between treatment and survival
- validate whether a proposed model recovers what it is supposed to estimate

This file should remain aligned with the model-development strategy.

### `survival_spatial_delta_only.py`

Touch this file only if there is a strong reason to adjust the current leading survival formulation.

At the moment, this file should be treated as the active survival baseline against which alternatives are judged.

### `treatment_spatial_pe.py`

Touch this file if you need to:

- refine the standalone treatment model
- add mild regularization or smoothing to the treatment baseline
- improve numerical behavior on treatment-only recovery

Any changes here should be validated against the existing treatment recovery workflow.

### `joint_spatial_treatment_survival.py`

This is the next major development file.

Touch this file if you need to:

- implement or refine the full joint model
- couple the treatment and survival submodels through spatial frailties
- expose deterministic nodes for joint-model diagnostics

### `run.py`

This is the shared inference wrapper layer.

Touch this file if you need to change:

- inference configuration defaults
- sample summarization logic
- standard handling of diagnostics such as divergences

### `scripts/`

Touch these files whenever a new model component needs validation. Every major change should come with a corresponding recovery or diagnostic script.

---

## 17. Practical development norms

A newcomer contributing to this project should follow these norms.

### 17.1 Preserve the data contract

New model files should consume the existing array contract whenever possible. Avoid creating one-off model-specific preprocessing branches unless there is a compelling reason.

### 17.2 Separate model changes from validation changes

When adding or modifying a model:

- first implement the model file
- then create or update a validation script
- then inspect both parameter recovery and latent-field recovery

### 17.3 Prefer simulation-aligned first versions

If there are multiple mathematically plausible formulations, begin with the one that matches the simulator. That gives the cleanest recovery target and the simplest debugging path.

### 17.4 Interpret hyperparameters cautiously

In spatial models of this kind, latent frailty fields are often better identified than the exact decomposition into mixing and scale hyperparameters. Prioritize effective frailty recovery over perfect hyperparameter recovery.

### 17.5 Do not overreact to tail instability without checking support

If a late interval behaves poorly, inspect interval support before changing the model. Low event counts and low effective risk-set size can explain noisy tail estimates even when the model is correct.

---

## 18. Current roadmap

The roadmap below reflects the current development trajectory.

### Phase 1: preserve the validated standalone components

Status: essentially complete.

Tasks:

- keep the survival component fixed on the delta-only formulation
- keep the standalone treatment component as the working treatment module
- preserve and maintain recovery scripts

### Phase 2: implement the first full joint model

Status: active next target.

Tasks:

- add `joint_spatial_treatment_survival.py`
- fit the joint model against the existing shared data contract
- recover:
  - survival fixed effects
  - treatment fixed effects
  - `delta_post`
  - `gamma`
  - `u_surv`
  - `u_ttt`
  - cross-process frailty coupling parameter
- begin with single-seed recovery
- then add multiseed joint recovery

### Phase 3: build prediction utilities

Status: next after basic joint-model recovery.

Tasks:

- build survival prediction utilities under alternative treatment times
- compute posterior survival curves by scenario
- compute time-point contrasts and RMST contrasts
- ensure the prediction interface is compatible with both the standalone survival fit and the future joint fit

### Phase 4: refine treatment and joint modeling only if diagnostics justify it

Status: deferred until needed.

Possible tasks:

- smooth the treatment baseline `gamma` if tail instability matters in practice
- strengthen priors on spatial hyperparameters if they destabilize joint fitting
- consider richer dependence structures only after the basic joint model is validated

### Phase 5: transition to real-data analysis

Status: downstream target.

Tasks:

- adapt graph construction to real SEER geography
- harden preprocessing for real cohort assembly
- define reporting utilities for substantive treatment-timing analysis
- build reproducible end-to-end analysis scripts

---

## 19. Suggested onboarding path for a new contributor

A new contributor should work through the project in the following order.

1. Read `prep.py` and understand the three time grids.
2. Read `model_data.py` and inspect the returned dictionary keys.
3. Read `simulate.py` to understand the truth-generating mechanism.
4. Read `survival_spatial_delta_only.py` to understand the current survival backbone.
5. Read the treatment recovery script and its outputs to understand how model validation is currently performed.
6. Only then begin modifying or implementing the joint model.

This order matters. Contributors who skip the preprocessing and simulation layers usually misunderstand the row semantics that drive the model definitions.

---

## 20. Immediate next tasks

If development resumes from the current state, the most logical next tasks are:

1. add the first full joint model file
2. write a single-seed joint recovery script
3. verify recovery of the cross-process frailty coupling
4. add a multiseed joint recovery script
5. begin designing prediction utilities for treatment-timing contrasts

That is the current best path forward. It builds directly on what has already been validated and avoids reopening parameterization questions that have effectively been settled.

