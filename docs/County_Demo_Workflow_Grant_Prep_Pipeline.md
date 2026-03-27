# County-Demo Workflow for the Grant-Prep Pipeline

This workflow documents the current Georgia county-based synthetic-data pipeline for the SEER-PEPH joint treatment-time / survival analysis.

## Purpose

The county-demo pipeline is intended to support:

- end-to-end pipeline testing
- grant-preparatory methods development
- realistic synthetic-data demonstrations
- future transition to county-based real-data analysis

The current baseline model is the frozen joint model with:

- piecewise-exponential treatment-time hazard
- piecewise-exponential survival hazard
- correlated spatial frailties
- linear-trend post-treatment survival effect

---

## Files

### Synthetic datasets

- `ga_crc_medicare_synth_10000.csv`
- `ga_crc_medicare_synth_25000.csv`

### Internal synthetic versions with truth fields

- `ga_crc_medicare_synth_10000_internal_with_truth.csv`
- `ga_crc_medicare_synth_25000_internal_with_truth.csv`

### County graph artifacts

- `ga_county_lookup.csv`
- `ga_county_graph_edges.csv`
- `ga_county_adjacency.json`
- `ga_county_graph_metadata.json`

### Analysis config

- `ga_joint_analysis_county_demo_config.json`

### Scripts / modules

- `src/seer_peph/analysis/joint_analysis.py`
- `scripts/run_joint_analysis.py`

---

## County-demo data design

The synthetic wide datasets include:

- subject identifier
- county FIPS and county name
- county-based `area_id`
- synthetic `zip` for backward compatibility
- demographic and clinical covariates
- treatment timing variables
- survival timing variables
- additional realistic fields for grant-demo purposes

The current analysis pipeline uses complete-case filtering on modeled covariates before long-format expansion.

---

## Graph mode

The county-demo workflow uses:

- `graph_mode = "county_graph_from_edges_file"`

The graph is built from the Georgia county edge list rather than a ring lattice.

This is the preferred mode for:

- synthetic Georgia county demonstrations
- grant-prep workflow testing
- future real county-based analyses

---

## Expected wide-form requirements

At minimum, the analysis input should include the columns needed by the joint config, such as:

- `id`
- `time`
- `event`
- `treatment_time`
- `treatment_time_obs`
- `treatment_event`
- `zip` or `area_id` or `county_fips`
- `sex`
- `stage`
- modeled covariates such as:
  - `age_per10_centered`
  - `cci`
  - `tumor_size_log`
  - `ses`

The county-demo synthetic datasets also include:

- `county_fips`
- `county_name`
- `area_id`

so the pipeline can prefer county-based area assignment directly.

---

## Run command

From the repository root:

```bash
python scripts/run_joint_analysis.py artifacts/ga_synth_grant_demo/ga_joint_analysis_county_demo_config.json
```

---

## What the runner does

The county-demo joint analysis runner:

1. loads the wide synthetic dataset
2. derives canonical internal columns if needed
3. drops rows with missing modeled covariates
4. builds the Georgia county graph from the edge file
5. constructs:
   - `surv_long`
   - `ttt_long`
6. fits the joint model
7. writes fit bundle, extraction summaries, and PPC outputs

---

## Main artifacts to inspect

### Data artifacts

- `wide_model_input.csv`
- `surv_long.csv`
- `ttt_long.csv`

### Fit artifacts

- `fit_bundle/`

### Survival summaries

- `joint_survival_beta_summary.csv`
- `joint_survival_alpha_summary.csv`
- `joint_survival_delta_post_summary.csv`
- `joint_survival_delta_post_linear_summary.csv`

### Treatment summaries

- `joint_treatment_theta_summary.csv`
- `joint_treatment_gamma_summary.csv`

### Spatial summaries

- `u_surv_summary.csv`
- `u_ttt_summary.csv`
- `u_ttt_ind_summary.csv`
- `joint_spatial_hyperparameter_summary.csv`

### Coupling summaries

- `joint_coupling_summary.csv`
- `joint_field_correlations_summary.csv`

### PPC summaries

- `joint_survival_ppc_interval_counts.csv`
- `joint_survival_ppc_area_counts.csv`
- `joint_survival_ppc_interval_by_treatment_counts.csv`
- `joint_treatment_ppc_interval_counts.csv`
- `joint_treatment_ppc_area_counts.csv`

### Metadata

- `analysis_config.json`
- `run_manifest.json`

---

## Current conventions

### Missing modeled covariates

Current behavior:
- drop observations with missing modeled covariates before model fitting

This is the current baseline convention for:
- smoke testing
- dev-scale runs
- grant-demo synthetic analyses

### Area unit priority

Current encoding logic prefers:

1. existing `area_id`
2. `county_fips`
3. `zip`

This allows the same pipeline to support both:
- county-based analysis
- legacy ZIP-based compatibility

---

## Recommended grant-prep usage

Use the county-demo workflow for:

- testing the full synthetic-to-analysis pipeline
- producing grant figures and workflow diagrams
- checking county-level PPC outputs
- validating county-graph ingestion
- rehearsing real-data analysis steps before cohort construction is finalized

Use the 10,000-row dataset for:
- rapid debugging
- script development
- smoke and dev runs

Use the 25,000-row dataset for:
- fuller demos
- more stable grant figures
- heavier workflow testing

---

## Next transition to real data

When moving from synthetic county-demo data to real cohort data, the next steps are:

1. build the real wide-form county-based cohort
2. preserve the same county graph interface
3. keep the same joint-analysis config structure where possible
4. rerun PPC, sensitivity checks, and interval-grid checks on the real data

This keeps the grant-prep synthetic workflow aligned with the eventual real-data pipeline.
