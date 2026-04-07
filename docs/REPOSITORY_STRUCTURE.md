# Repository Structure

This document lists the main directories and files in the repository.

## Top-Level Layout

```text
neural-edge-video-compression/
|-- README.md
|-- run_compression.py
|-- run_decompression.py
|-- configs/
|-- src/
|-- scripts/
|-- tests/
|-- docker/
|-- _third_party_amt/
|-- data/      # local, gitignored except .gitkeep
|-- models/    # local, gitignored except .gitkeep
`-- outputs/   # local, gitignored except .gitkeep
```

## Main Files And Directories

### `README.md`

Project overview, setup steps, run commands, and result summary.

### `run_compression.py`

Top-level compression entry point. Loads the compression config, runs ROI generation and frame selection, encodes the streams, and writes the archive.

### `run_decompression.py`

Top-level decompression entry point. Reads the archive, reconstructs the sparse streams, restores missing frames, and writes the output video.

### `configs/`

Versioned runtime profiles.

Important files:

- `configs/gpu/compression.yaml`
- `configs/gpu/decompression.yaml`

### `src/`

Main implementation organized by pipeline stage.

Current stage structure:

- `src/roi_detection/`
- `src/frame_removal/`
- `src/compression/`
- `src/decompression/`
- `src/pipeline/`

### `scripts/`

Sanity-check helpers for ROI detection, frame removal, compression, decompression, and model download.

### `tests/`

Automated checks for config handling, stage logic, runtime contracts, archive structure, and deterministic behavior where expected.

### `docker/`

Containerized GPU runtime path.

### `_third_party_amt/`

Vendored interpolation dependency.

## Documentation Files

- `docs/RESULTS_SUMMARY.md`
- `docs/project_summary.md`
- `docs/dataset_and_eval.md`
- `docs/reproducibility_validation.md`
- `docs/figures/README.md`
