# SEER-PEPH Project Roadmap: From Current State to Real-Data-Ready Applications Analysis

## Current state

The project has moved beyond prototyping. You now have a coherent end-to-end modeling pipeline with validated standalone components and a viable first joint model.

### What is already in place

You have working modules for:

- preprocessing
- graph construction
- simulation
- model-data assembly
- inference wrappers

On the modeling side, you now have:

- a standalone **survival model**
  - spatial piecewise-exponential
  - baseline covariates
  - delta-only post-treatment effects
  - BYM2-style spatial frailty
- a standalone **treatment-time model**
  - spatial piecewise-exponential
  - baseline covariates
  - BYM2-style spatial frailty
- a first **joint model**
  - current survival model
  - current treatment model
  - cross-process coupling through correlated area-level frailties

### What has been validated

The survival model is reasonably well validated for the intended use case. The delta-only parameterization is the right working specification. The older `beta_td + delta_post` formulation is no longer the preferred route.

The treatment model also looks good enough to be considered a stable standalone module. The main weakness is expected tail instability in late treatment baseline intervals, and the interval-support diagnostics suggest that this is driven by data support rather than clear misspecification.

The joint model has passed the smoke-test stage and a multi-seed recovery stage. The fixed-effect blocks are mostly stable, the cross-process spatial association is estimable, and the latent spatial fields recover well. The weakest parts remain:

- some hyperparameter decomposition
- moderate attenuation in parts of `delta_post`
- persistent attenuation of the tumor-size survival coefficient

That tumor-size issue appears to be inherited from the standalone survival model, not introduced by the joint model.

### What is not yet ready for real-data analysis

The package is not yet ready for an applications paper because several things are still missing or only partially developed:

- prediction utilities for survival under alternative treatment times
- a stable user-facing fit/predict/report interface
- formal posterior predictive checks and real-data diagnostics
- robust handling of preprocessing metadata and analysis specifications
- model comparison and sensitivity-analysis workflows
- production-quality reporting outputs for paper figures and tables
- documentation of default analysis workflow for real SEER-Medicare data

So the project is in a strong **method-development / internal-validation** state, but not yet in a **real-data analysis / paper-production** state.

---

## Main strategic goal

The next goal should not be “add more model ideas.”

It should be:

**freeze the first scientifically usable model stack, build prediction and diagnostics around it, and harden the pipeline so it can support a real-data analysis transparently and reproducibly.**

That means the center of gravity should shift from parameterization exploration to:

- interface design
- diagnostics
- prediction
- reproducibility
- analysis workflow

---

## Recommended target for the applications-paper version

I would target the following as the first real-data-ready model stack:

### Core model stack

1. **Standalone survival model**
   - current delta-only survival model
   - remains the reference survival component

2. **Standalone treatment model**
   - current treatment PE model
   - remains the reference treatment component

3. **Joint model v1**
   - current joint spatial treatment-survival model
   - correlated area frailty coupling
   - no extra subgroup interactions yet
   - no extra nonlinear structure yet

### Primary inferential outputs

For the paper, the package should be able to produce:

- covariate effects on treatment timing
- covariate effects on survival
- post-treatment hazard trajectory estimates
- spatial residual maps for treatment and survival
- estimated association between residual treatment-delay geography and residual survival geography
- counterfactual survival curves under different treatment initiation times

That last item is the most important missing capability right now.

---

## Roadmap from current state to real-data-ready

I would organize the remaining work into six phases.

# Phase 1. Freeze the current validated model stack

Goal: stop reopening core model design questions unless diagnostics force it.

### Tasks

1. Freeze the current default models:
   - `survival_spatial_delta_only`
   - `treatment_spatial_pe`
   - `joint_spatial_treatment_survival`

2. Mark known caveats explicitly:
   - tumor-size attenuation in survival
   - weaker treatment baseline recovery in late intervals
   - hyperparameters less trustworthy than latent fields

3. Define the official modeling contract:
   - expected input columns
   - graph structure requirements
   - long-format conventions
   - covariate ordering
   - posterior sample naming conventions

4. Add a small set of regression tests that fail if the model interfaces drift.

### Acceptance criteria

- no ambiguity about which model files are “active”
- no ambiguity about covariate order or parameter meaning
- the current recovery scripts run without ad hoc edits

This phase is mostly organizational, but it matters. Without it, later paper work becomes fragile.

---

# Phase 2. Build a stable fit object and package interface

Goal: move from raw script-level fitting to a usable package API.

Right now, a lot of the workflow still looks like “construct data dict, call model, summarize samples.” That is fine for development, but not for real analysis.

### Tasks

1. Create stable wrappers like:
   - `fit_survival_model(...)`
   - `fit_treatment_model(...)`
   - `fit_joint_model(...)`

2. Return structured fit objects that store:
   - posterior samples
   - posterior summary
   - model type
   - covariate names
   - break grids
   - scaling metadata
   - config used for fitting
   - graph metadata

3. Standardize extraction helpers:
   - `extract_survival_effects(...)`
   - `extract_treatment_effects(...)`
   - `extract_spatial_fields(...)`
   - `extract_joint_coupling(...)`

4. Add model serialization:
   - save fit objects cleanly
   - reload fit objects without recomputation

### Acceptance criteria

- real-data analysis can be run without touching model internals
- scripts no longer need to know parameter index positions by hand
- downstream prediction and reporting functions can consume fit objects directly

This is the single most important software-engineering step remaining.

---

# Phase 3. Implement prediction utilities

Goal: make the models scientifically useful for treatment-timing analysis.

This is the biggest substantive gap.

## 3A. Survival prediction from the standalone survival model

You need utilities that can compute:

- hazard paths
- survival curves
- RMST
- survival contrasts at fixed times

under alternative treatment initiation scenarios.

### Tasks

1. Build low-level functions:
   - piecewise hazard construction from posterior draws
   - survival curve construction from hazards
   - scenario evaluation for treatment at time `t*`

2. Build posterior predictive wrappers:
   - draw-level predictions
   - posterior means and credible bands

3. Build contrast utilities:
   - treatment at 1 month vs 3 months
   - treatment at 3 months vs 6 months
   - no treatment vs observed-treatment scenario
   - RMST differences

### Acceptance criteria

- can produce subject-level or profile-level survival curves under alternative treatment times
- can summarize clinically interpretable contrasts for paper figures

## 3B. Prediction from the joint model

Once the standalone survival prediction machinery exists, extend it to the joint model.

Initially, I would keep this modest:
- use posterior draws from the joint model
- compute the same survival scenario outputs
- do not yet attempt a full causal estimand framework

### Acceptance criteria

- joint-model prediction preserves the same output contract as survival-only prediction
- paper-ready plots can be generated from joint posterior draws

---

# Phase 4. Build a real-data analysis workflow

Goal: go from “model code” to “analysis pipeline.”

This phase is where the package becomes genuinely usable for the SEER-Medicare application.

### Tasks

1. Define the real-data analysis dataset contract:
   - one row per patient
   - required fields
   - coding conventions
   - censoring definitions
   - treatment-event definitions
   - county/ZIP graph linkage rules

2. Build a top-level preparation function:
   - validate raw analysis input
   - encode categorical variables
   - derive time variables
   - construct area IDs
   - apply any transformations
   - create survival and treatment long tables
   - store preprocessing metadata

3. Build a single analysis driver:
   - read config
   - prepare data
   - fit model
   - save outputs
   - generate diagnostics
   - generate paper-ready summaries

4. Add deterministic artifact structure:
   - `data/processed/...`
   - `artifacts/model_fit/...`
   - `artifacts/diagnostics/...`
   - `artifacts/figures/...`
   - `artifacts/tables/...`

### Acceptance criteria

- a newcomer can run a full real-data model fit from one config
- outputs land in predictable places
- reruns are reproducible

---

# Phase 5. Add diagnostics for real-data credibility

Goal: ensure the model is usable scientifically, not just computationally.

Internal recovery is necessary but not sufficient. For a paper, you need model-checking tools.

### Tasks

## 5A. Posterior predictive checks

For survival:
- interval event counts
- cumulative event curves
- survival by important strata
- observed vs posterior-predicted hazards

For treatment:
- treatment-start curves
- interval event counts
- observed vs predicted treatment timing distribution

For joint structure:
- geographic residual maps
- correlation of fitted treatment and survival frailty surfaces
- observed vs fitted area-level summaries

## 5B. Sensitivity analyses

Minimum set:
- prior sensitivity
- graph sensitivity
- interval-grid sensitivity
- treatment definition sensitivity
- covariate inclusion sensitivity

You do not need to explore everything. But the package should make these easy.

## 5C. Stability diagnostics

- sampler diagnostics
- ESS / R-hat summaries
- divergence summaries
- runtime and memory logs

### Acceptance criteria

- every real-data fit automatically produces a core diagnostic report
- the package can support the “robustness” section of an applications paper

---

# Phase 6. Build paper-oriented outputs

Goal: turn model fits into publishable results.

The package should be able to produce, with minimal manual intervention:

### Tables

- baseline covariate effect tables
- treatment-model effect tables
- post-treatment effect tables
- spatial hyperparameter and coupling summaries
- sensitivity-analysis tables

### Figures

- treatment-start curves
- counterfactual survival curves
- post-treatment effect trajectory
- county-level residual maps
- observed vs fitted calibration plots
- key posterior contrast plots

### Narrative summaries

- standardized summary text for major results
- logging of exact model version and config used

### Acceptance criteria

- the package can generate the core tables and figures for a draft paper
- rerunning the same config regenerates the same outputs

---

## Priority order from here

If the goal is “usable for an applications paper,” I would do the remaining work in this order:

### Priority 1
Freeze interfaces and build fit objects.

### Priority 2
Implement counterfactual survival prediction.

### Priority 3
Build top-level real-data preparation and model-run drivers.

### Priority 4
Add posterior predictive checks and reporting outputs.

### Priority 5
Do targeted sensitivity analyses.

### Priority 6
Only then consider model enrichments like rural/urban interactions, nonlinear tumor effects, or alternative priors.

That order matters. Right now the risk is spending more time refining model structure when the biggest missing pieces are usability and interpretation.

---

## What not to do yet

I would avoid these until the core pipeline is paper-ready:

- more exploratory parameterizations of the survival model
- richer joint dependence structures
- subgroup-specific treatment-effect interactions
- nonlinear tumor-size modeling
- causal-estimand language beyond counterfactual scenario prediction

Those are all worthwhile later, but they can easily delay the point at which you have a usable analysis system.

---

## Concrete next milestone

The next milestone I would define is:

**Milestone A: first real-data-ready analysis pipeline using the current joint model with posterior prediction and core diagnostics.**

To hit that milestone, the immediate next tasks are:

1. build fit-object wrappers
2. implement prediction utilities
3. build the top-level analysis runner
4. generate a standard diagnostic/report bundle

Once that milestone is complete, the package becomes practically usable for the applications paper.

---

## Suggested deliverables for the next 4 work blocks

### Block 1: interface hardening
- fit wrappers
- structured fit objects
- save/load utilities
- extraction helpers

### Block 2: prediction
- survival scenario prediction
- RMST and survival contrast summaries
- plotting helpers

### Block 3: real-data workflow
- preprocessing driver
- fit driver
- config schema for real analyses
- artifact layout

### Block 4: paper outputs
- posterior predictive checks
- standard tables
- standard figures
- analysis summary report

---

## Final assessment

The project is in a strong place.

You now have:
- a stable survival model
- a stable treatment model
- a viable joint model
- internal recovery evidence that the core structure works

So the project’s bottleneck is no longer “find the right model.”  
The bottleneck is now “turn the validated model into a robust, interpretable, reproducible analysis system.”

That is a good place to be.

The single best next move is to start building the fit/predict/report interface around the current joint model rather than reopening core model design.
