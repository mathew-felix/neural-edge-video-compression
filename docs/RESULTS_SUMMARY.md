# Results Summary

This file gives a thesis-facing summary of the main reported results so readers can understand the project without digging through raw experiment folders first.

## Main Thesis Question

Can wildlife video be compressed in a way that reduces transmission and storage cost substantially while preserving the animal region better than treating the whole frame uniformly?

## Reported Pipeline Highlights

Released-pipeline summary on the held-out test set:

- original test set size: `399.16 MB`
- transmitted archive size: `10.36 MB`
- transmitted-size reduction: `97.4%`
- aggregate compression ratio: `38.52x`
- mean ROI PSNR: `36.12 dB`
- mean ROI MS-SSIM: `0.9758`
- mean downstream detector proxy recall: `79.11%`

These numbers support the thesis-level claim that the transmitted result can be reduced substantially while preserving the animal region more faithfully than the frame as a whole.

The defended thesis also reports LPIPS, but the public GitHub summary intentionally emphasizes size reduction, ROI PSNR, ROI MS-SSIM, and downstream utility instead of LPIPS-based figures.

## ROI Generation Highlight

Compared with dense detection:

- detector calls reduced by `93.28%`
- ROI-stage speed improved by `4.41x`
- ROI-presence recall remained at `96.24%`
- frame-level agreement remained at `95.25%`

This supports the use of sparse detection plus propagation under resource constraints.

## Claim Boundary

This repository does not claim that the proposed method is universally better than every baseline on every metric.

The defended thesis claim is narrower:

- ROI-priority processing can reduce transmitted size substantially
- sparse ROI generation is accurate enough for the downstream stages
- the animal region can be preserved better than the frame as a whole
- the system is meaningful to discuss under edge-oriented constraints

## Best Way To Use This File

Use this page when you need to:

- explain the project quickly to recruiters
- link a short results page from LinkedIn or a resume
- orient a thesis committee reader before they inspect code
