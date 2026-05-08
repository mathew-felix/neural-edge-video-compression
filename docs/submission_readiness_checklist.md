# Submission Readiness Checklist

## Claims and Scope

- [ ] Main claim uses only measured results
- [ ] No Jetson performance claims
- [ ] DGX encode + laptop decode flow is explicitly stated

## Experiment Integrity

- [ ] Clip manifest is frozen and versioned
- [ ] Baselines are run with matched bitrate budget
- [ ] Ablations are minimal and decision-relevant
- [ ] Results schema is complete in `outputs/paper_runs/results.csv`

## Reproducibility

- [ ] README run commands are correct
- [ ] Runtime versions are pinned and documented
- [ ] Model checksums verified using `docs/model_checksums.sha256`
- [ ] Fresh-run smoke test succeeds from clean environment

## Writing and Figures

- [ ] All main tables generated from logged outputs
- [ ] Failure cases are included with explanations
- [ ] Limitations section is explicit and honest
- [ ] Appendix includes exact config references
