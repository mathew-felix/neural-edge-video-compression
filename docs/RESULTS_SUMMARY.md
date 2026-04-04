# Results Summary

## Main Question

Can wildlife video be compressed in a way that reduces transmission and storage cost substantially while preserving the animal region better than treating the whole frame uniformly?

## Held-Out Test Set

Reported summary on the held-out test set:

- original test set size: `399.16 MB`
- transmitted archive size: `10.36 MB`
- transmitted-size reduction: `97.4%`
- aggregate compression ratio: `38.52x`
- mean ROI PSNR: `36.12 dB`
- mean ROI MS-SSIM: `0.9758`
- mean downstream detector proxy recall: `79.11%`

The thesis also reports LPIPS. The repo summary files focus on size reduction, ROI PSNR, ROI MS-SSIM, and downstream detector utility.

## ROI Generation Stage

Compared with dense detection:

- detector calls reduced by `93.28%`
- ROI-stage speed improved by `4.41x`
- ROI-presence recall: `96.24%`
- frame-level agreement: `95.25%`

## Scope

This repository does not claim that the method is better than every baseline on every metric.

The reported result is narrower:

- ROI-priority processing can reduce transmitted size substantially
- sparse ROI generation is accurate enough for the downstream stages
- the animal region can be preserved better than the frame as a whole
- the system operates within the tested edge-oriented configuration
