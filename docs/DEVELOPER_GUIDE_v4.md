# SEER-PEPH Developer Guide

Updated for the current implementation state of the SEER-PEPH treatment-timing / survival modeling project.

## 1. Purpose and scope

SEER-PEPH is a research package for building, validating, and ultimately applying Bayesian spatial treatment-time and survival models to SEER-Medicare-style cancer data. The current code base supports three closely related analysis tracks:

- standalone survival modeling
- standalone treatment-time modeling
- joint treatment-survival modeling

The package is designed around a few stable principles:

1. wide subject-level data are converted into explicit long-format piecewise-exponential representations
2. model files consume a stable array/data contract rather than raw pandas logic
3. every major model component is developed through simulation and recovery before being trusted on real data
4. analysis should be reproducible through config-driven runners rather than ad hoc notebooks

The end goal is an applications-paper workflow that can answer treatment-timing questions with transparent diagnostics, structured validation, and reproducible artifacts.

---

## 2. Current project status

At the current stage, the following pieces are implemented and working end to end.

### Data and preprocessing
- wide-to-long preprocessing for survival and treatment processes
- configurable interval grids
- configurable raw input schema and derived-column schema
- stable long-format semantics for survival and treatment risk sets

### Modeling and fitting
- standalone spatial delta-only survival model
- standalone spatial treatment-time piecewise-exponential model
- full joint treatment-survival model fitted through `fit_joint_model(...)`
- fit wrappers for survival, treatment, and joint models
- fit-object save/load support

### Post-fit utilities
- extraction helpers for survival, treatment, spatial, and joint-coupling summaries
- standalone survival prediction utilities
- survival contrast utilities for treatment-timing comparisons
- posterior predictive checks for:
  - standalone survival
  - standalone treatment
  - joint model, separately for survival and treatment components

### Analysis runners
- standalone survival analysis runner and CLI
- standalone treatment analysis runner and CLI
- joint analysis runner and CLI

### Validation framework
- scenario-driven model-matched joint simulator
- structured simulation scenario and result objects
- single-seed joint validation runner
- multi-scenario, multi-seed joint validation study runner
- config-driven validation study execution
- recovery summaries and study-level aggregation

The project is no longer at the stage where the joint model is merely planned. It is implemented, runnable, and under formal simulation-based validation. The main work now is strengthening validation, diagnostics, and real-data readiness.

---

## 3. High-level architecture

The code base is now organized around five layers.

### 3.1 Preprocessing layer
Responsible for:
- loading/encoding wide data
- constructing canonical modeling columns
- expanding to survival and treatment long formats
- enforcing interval/event semantics

Main module:
- `seer_peph.data.prep`

### 3.2 Model-data / fitting layer
Responsible for:
- assembling model inputs
- fitting survival, treatment, and joint models
- storing fit metadata
- saving and reloading fit bundles
- extracting posterior summaries

Main modules:
- `seer_peph.fitting.fit_models`
- `seer_peph.fitting.io`
- `seer_peph.fitting.extract`
- `seer_peph.inference.run`

### 3.3 Analysis layer
Responsible for:
- config-driven model runs
- artifact writing
- prediction summaries
- PPC artifact generation
- top-level analysis manifests

Main modules:
- `seer_peph.analysis.survival_analysis`
- `seer_peph.analysis.treatment_analysis`
- `seer_peph.analysis.joint_analysis`

### 3.4 Diagnostics / prediction layer
Responsible for:
- survival prediction and contrasts
- posterior predictive checks
- model adequacy summaries

Main modules:
- `seer_peph.predict.survival`
- `seer_peph.predict.survival_contrasts`
- `seer_peph.diagnostics.survival_ppc`
- `seer_peph.diagnostics.treatment_ppc`
- `seer_peph.diagnostics.joint_ppc`

### 3.5 Validation layer
Responsible for:
- structured simulation scenarios
- structured truth bundles
- single-seed validation runs
- multi-seed study aggregation

Main modules:
- `seer_peph.validation.joint_scenarios`
- `seer_peph.validation.joint_results`
- `seer_peph.validation.simulate_joint`

---

## 4. Current data-flow design

The current package follows this conceptual pipeline.

1. Start with a subject-level wide dataframe.
2. Encode or canonicalize required modeling columns:
   - `time_m`
   - `treatment_time_m`
   - `treatment_time_obs_m`
   - dummy-coded covariates such as `sex_male`, `stage_II`, `stage_III`
   - `area_id`
3. Expand the wide data into:
   - `surv_long`
   - `ttt_long`
4. Fit one of:
   - survival model
   - treatment model
   - joint model
5. Save the fit bundle and write summary artifacts.
6. Optionally compute:
   - survival predictions / contrasts
   - PPC tables
   - validation recovery tables

The key design point is that model-fitting code does not operate directly on arbitrary raw wide data. The analysis runners first canonicalize the schema, then expand to long format, then fit.

---

## 5. Preprocessing layer: current contract

The preprocessing layer is still the most important part of the code base to understand.

### Core time grids
There are three distinct grids:

- `SURV_BREAKS`: baseline survival intervals
- `TTT_BREAKS`: baseline treatment-time intervals
- `POST_TTT_BREAKS`: post-treatment intervals used only in the survival model

These play different roles and should not be conflated.

### Survival long-format semantics
Each survival long row contains:
- `id`
- `k`
- `t0`, `t1`
- `exposure`
- `event`
- `area_id`
- modeled covariates
- `treated_td`
- `k_post`

A row is cut so that it lies entirely within:
- one survival baseline interval
- one treatment-status regime
- one post-treatment interval if treated

### Treatment long-format semantics
Each treatment long row contains:
- `id`
- `k`
- `t0`, `t1`
- `exposure`
- `event`
- `area_id`
- modeled covariates

Treatment follows semi-competing-risks semantics: death censors treatment follow-up. This is explicit in the current `build_treatment_long(...)` contract.

### Configurable schemas
The analysis runners now support both:
- configurable raw input column names
- configurable derived/model-ready column names

So the code no longer assumes that input data always use the package’s default field names. The runner-level schema config is now part of the public architecture.

---

## 6. Current model inventory

### 6.1 Standalone survival model
The leading survival model remains the delta-only spatial survival model.

Its practical form is:

\[
\log \lambda^{surv}_r
=
\alpha_{k_r}
+
x_r^\top \beta
+
u_{a_r}
+
\text{treated}_r \, \delta_{k_{post,r}}
\]

Key points:
- no separate `beta_td` term
- treatment-history effect represented directly through `delta_post`
- spatial frailty included through a BYM2-style structure
- current survival development is centered on this parameterization

### 6.2 Standalone treatment model
The standalone treatment model is a piecewise-exponential treatment-time hazard model:

\[
\log \lambda^{ttt}_q
=
\gamma_{k_q}
+
z_q^\top \theta
+
u^{ttt}_{a_q}
\]

Key points:
- baseline treatment intervals indexed by `gamma[k]`
- treatment covariate effects `theta`
- spatial frailty `u_ttt`
- currently no smoothing prior on `gamma` by default

### 6.3 Joint model
The joint model is no longer just a design target. It is part of the working code path.

Its intended structure is still the same as the earlier design:
- survival submodel
- treatment submodel
- cross-process dependence through correlated spatial frailties

Conceptually:

\[
\log \lambda^{surv}_r
=
\alpha_{k_r}
+
x_r^\top \beta
+
u^{surv}_{a_r}
+
\text{treated}_r \, \delta_{k_{post,r}}
\]

\[
\log \lambda^{ttt}_q
=
\gamma_{k_q}
+
z_q^\top \theta
+
u^{ttt}_{a_q}
\]

with cross-process coupling represented through correlated latent spatial fields, matching the simulation framework.

The current development stance is:
- the first joint model should remain close to the simulator
- unnecessary additional dependence structures should be avoided until validation justifies them

---

## 7. Fitting layer and fit objects

The fitting layer now has a much more formal interface than in earlier stages.

### Fit wrappers
Current wrapper functions include:
- `fit_survival_model(...)`
- `fit_treatment_model(...)`
- `fit_joint_model(...)`

These standardize:
- model invocation
- metadata tracking
- inference configuration
- fit object structure

### Save/load support
Fit bundles can now be persisted and reloaded. The fit I/O layer writes:
- fit manifest
- posterior samples
- serialized data arrays
- summaries
- scalar summary tables

This matters because analysis runners and validation scripts can now produce self-contained fit artifacts rather than only transient in-memory outputs.

### Extraction helpers
Extraction utilities now provide standardized summary tables for:
- survival effects
- treatment effects
- spatial fields
- joint coupling summaries

This removes a lot of ad hoc summary logic from scripts and makes runner outputs more uniform.

---

## 8. Analysis runner layer

One of the biggest changes since the earlier guide is that the package now has proper config-driven analysis runners.

### 8.1 Standalone survival analysis
`seer_peph.analysis.survival_analysis` now supports:
- config-driven schema handling
- config-driven breaks
- fit/save/load
- survival summary artifacts
- survival predictions
- survival contrasts
- survival PPC artifacts

### 8.2 Standalone treatment analysis
`seer_peph.analysis.treatment_analysis` now supports:
- config-driven schema handling
- config-driven treatment covariates
- fit/save/load
- treatment summary artifacts
- treatment PPC artifacts

### 8.3 Joint analysis
`seer_peph.analysis.joint_analysis` now supports:
- config-driven schema handling
- configurable survival and treatment covariate sets
- fit/save/load
- survival/treatment/spatial/coupling artifact writing
- joint PPC artifact writing

### CLI runners
There are now top-level scripts for:
- `run_survival_analysis.py`
- `run_treatment_analysis.py`
- `run_joint_analysis.py`

These are intended to be the primary reproducible entry points for actual analyses.

---

## 9. Prediction and contrast utilities

This is another major update relative to the earlier guide.

### Current scope
The package now includes standalone survival prediction utilities for:
- survival probabilities at landmark times
- RMST summaries
- survival contrasts under alternative treatment times
- RMST contrasts under alternative treatment times

These are currently the most mature paper-facing post-fit utilities in the package.

### Current limitation
Equivalent paper-facing joint prediction utilities are not yet as mature as the standalone survival prediction stack. That remains one of the main next steps for substantive analysis readiness.

---

## 10. Posterior predictive checks

The package now includes first-pass PPC infrastructure for all three analysis tracks.

### Standalone survival PPCs
Current summaries include:
- observed vs posterior-predicted survival counts by interval
- observed vs posterior-predicted survival counts by area
- observed vs posterior-predicted survival counts by interval × treatment status

### Standalone treatment PPCs
Current summaries include:
- observed vs posterior-predicted treatment counts by interval
- observed vs posterior-predicted treatment counts by area

### Joint PPCs
The joint fit currently produces both:
- survival PPCs under the joint posterior
- treatment PPCs under the joint posterior

These are intentionally count-based first-pass diagnostics. They are meant to answer whether the fitted model reproduces the major observed event patterns before moving on to more elaborate residual-style or graphical diagnostics.

---

## 11. Validation framework: current state

This is the largest architectural change since the earlier guide.

The earlier guide described simulation and recovery as the next development step. That is no longer accurate. The project now has a more formal validation framework.

### 11.1 Scenario-driven simulation
Validation is now based on:
- `JointSimulationScenario`
- `JointSimulationResult`

These make the DGP explicit and structured rather than relying on a long loose argument list.

A scenario defines:
- graph structure
- sample size
- interval grids
- survival/treatment baselines
- fixed effects
- spatial parameters
- cross-process coupling
- censoring regime
- covariate-generation controls

### 11.2 Structured simulation results
A simulation replicate returns:
- wide simulated data
- parameter truth table
- area-level truth table
- support diagnostics
- replicate metadata

This is a much cleaner validation interface than embedding truth only as wide-data columns.

### 11.3 Model-matched validation study runners
There are now scripts for:
- single-seed joint validation
- full multi-seed study execution

The current study runner supports:
- config-driven scenario lists
- config-driven seed lists
- config-driven inference settings
- per-seed artifact writing
- aggregated study summaries

### 11.4 Recovery summaries
The current validation study writes structured summaries for:
- survival beta recovery
- treatment theta recovery
- survival alpha recovery
- treatment gamma recovery
- post-treatment delta recovery
- area-level field recovery
- fit diagnostics

This is the main current mechanism for judging whether the joint implementation is working correctly under model-matched truth.

---

## 12. Current interpretation of validation status

The project is now in a stronger position than the earlier guide implied, but not all parts are equally mature.

### Stable or close to stable
- preprocessing semantics
- standalone survival analysis path
- standalone treatment analysis path
- fit bundle save/load
- extraction helpers
- first-pass PPC infrastructure
- config-driven analysis runners

### Functioning but still under active validation
- full joint model
- formal model-matched joint validation study
- interpretation of weaker parameter blocks such as some `delta_post` components or harder-to-identify latent fields

### Not yet mature enough to treat as finished
- paper-facing joint prediction/report stack
- real-data audit / cohort-summary runner
- misspecified robustness validation tier
- real-data sensitivity-analysis automation

---

## 13. What should currently be treated as “frozen enough”

A contributor joining now should treat these as the current defaults unless a specific validation result says otherwise:

- survival model remains the delta-only treatment-history formulation
- treatment model remains a spatial PE treatment-time model with independent `gamma[k]`
- config-driven analysis runners are the preferred execution path
- scenario-driven validation is the preferred model-validation path
- joint validation should begin model-matched before robustness checks

The earlier recommendation not to reopen the old `beta_td + delta_post` survival parameterization still stands.

---

## 14. What remains the highest-priority development work

The next high-priority tasks are no longer core plumbing. They are analysis-readiness tasks.

### 14.1 Real-data audit layer
Before real-data fitting, the package still needs a formal audit/cohort-summary runner that reports:
- missingness
- support by interval
- area-level counts
- censoring structure
- treatment-incidence structure
- graph sanity checks

### 14.2 Stronger validation summaries
The validation framework now exists, but still needs:
- broader model-matched runs
- divergence tracking in study summaries
- eventually a misspecified robustness tier

### 14.3 Joint paper-facing outputs
The package still needs stronger joint-model utilities for:
- treatment-timing contrasts
- RMST contrasts
- subgroup contrasts
- area-level substantive summaries

### 14.4 Real-data sensitivity workflow
For applications-paper readiness, the package still needs a planned set of sensitivity analyses:
- grid sensitivity
- covariate sensitivity
- treatment-effect parameterization sensitivity if needed
- prior sensitivity
- graph sensitivity if real adjacency choices vary

---

## 15. Recommended development norms

### Preserve the preprocessing contract
Changes to row semantics belong in preprocessing, not scattered across models or scripts.

### Prefer config-driven runners over ad hoc scripts
The new analysis and validation runners should be the primary execution path whenever possible.

### Validate every model change through simulation
Any meaningful change to the joint model should be checked through:
- a single-seed dry run
- a mini-pilot
- then a broader study

### Distinguish coding failure from model difficulty
When recovery is weak:
- first check whether the issue persists in easy model-matched scenarios
- then inspect support and computation
- only then decide whether the issue is implementation or identification

### Do not overinterpret hyperparameter weakness
As before, latent field recovery is generally more important than exact hyperparameter recovery in these spatial models.

---

## 16. Suggested onboarding path for a new contributor

A newcomer should now read the project in this order:

1. `seer_peph.data.prep`
2. `seer_peph.analysis.survival_analysis`
3. `seer_peph.analysis.treatment_analysis`
4. `seer_peph.analysis.joint_analysis`
5. `seer_peph.fitting.fit_models`
6. `seer_peph.fitting.extract`
7. `seer_peph.validation.joint_scenarios`
8. `seer_peph.validation.simulate_joint`
9. `scripts/run_joint_validation_seed.py`
10. `scripts/run_joint_validation_study.py`

The older emphasis on a future joint-model layer should now be replaced with the already-implemented scenario-driven validation modules and analysis runner stack.

---

## 17. Immediate next tasks

If development resumes from the current state, the most logical next tasks are:

1. finish divergence capture and study-level computational diagnostics in the validation summaries
2. complete the broader model-matched joint validation study
3. add the first misspecified robustness validation tier
4. build a real-data audit / cohort-summary runner
5. extend the joint-model post-fit utilities toward paper-facing treatment-timing contrasts
6. define the frozen real-data analysis specification for the applications paper

That is the shortest path from the current code base to a real-data analysis workflow that is scientifically defensible and reproducible.
