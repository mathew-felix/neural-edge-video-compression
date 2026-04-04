# Repository Structure

This document explains how the repository is organized and what each area is responsible for.

## Top-Level Layout

```text
neural-edge-video-compression/
├── README.md
├── run_compression.py
├── run_decompression.py
├── configs/
├── src/
├── scripts/
├── tests/
├── docker/
├── DCVC/
├── _third_party_amt/
├── data/            # local, gitignored
├── models/          # local, gitignored
└── outputs/         # local, gitignored
```

## What Each Area Should Contain

### `README.md`

The public front door of the project.

It should answer:

- what the project is
- why it matters
- what the pipeline does
- how to run it
- what the claim boundaries are

### `run_compression.py`

Top-level compression entry point.

Responsible for:

- loading the compression config
- running ROI generation and frame selection
- invoking learned compression
- writing the transmitted archive

### `run_decompression.py`

Top-level decompression entry point.

Responsible for:

- reading the transmitted archive
- restoring sparse anchors
- reconstructing the final video
- writing the reconstructed output

### `configs/`

Versioned runtime profiles.

Current important files:

- `configs/gpu/compression.yaml`
- `configs/gpu/decompression.yaml`

For a thesis-facing artifact, configs should stay stable and readable.

### `src/`

Main project implementation organized by pipeline stage.

Current stage structure:

- `src/roi_detection/`
- `src/frame_removal/`
- `src/compression/`
- `src/decompression/`
- `src/pipeline/`

This is a good high-level layout because it matches the thesis narrative.

### `scripts/`

Developer and artifact sanity-check helpers.

These should stay lightweight and stage-specific.

They are useful for:

- quick smoke tests
- visual inspection
- reproducibility checks

They should not become a second, competing public interface.

### `tests/`

Automated checks for:

- config schema behavior
- stage logic
- runtime contracts
- archive structure
- deterministic behavior where expected

For a thesis repo, tests are credibility multipliers.

### `docker/`

Containerized GPU runtime path.

This should exist for:

- reproducible setup
- avoiding local dependency chaos
- artifact-style evaluation environments

### `DCVC/`

Vendored compression dependency.

Keep it clearly marked as third-party.

### `_third_party_amt/`

Vendored interpolation dependency.

Keep it clearly marked as third-party.

## Recommended Documentation Additions

To make the repository stronger for thesis, jobs, and PhD review, the most valuable additions are:

1. `docs/RESULTS_SUMMARY.md`
2. `docs/dataset_and_eval.md`
3. `docs/figures/`
4. `docs/limitations.md`
5. `docs/release_artifact_checklist.md`

The repository now includes `docs/figures/` for the public GitHub figure set. That folder should stay aligned with the thesis while avoiding LPIPS-based charts in the public-facing presentation.

## What The Repository Should Feel Like

An external reader should feel:

- this project has a clear scientific purpose
- the code structure matches the thesis stages
- the public entry points are obvious
- dependencies are understandable
- the repository is organized enough to extend

It should not feel like:

- a private experiment dump
- a folder of ad hoc scripts
- a repo that only the original author can navigate
