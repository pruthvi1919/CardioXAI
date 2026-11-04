Data: SYNTHETIC test images (real MIT-BIH test beats) | 60 images per condition | image 256x768
Metric: correlation of each digitizer's trace with the oracle trace from the ground-truth mask.
Resolution ceiling (oracle trace vs drawn signal, 200 pts): r = 0.978

| Condition | Classical mean r | Classical % r>0.95 | U-Net mean r | U-Net % r>0.95 |
|---|---|---|---|---|
| clean | 0.754 | 63% | 0.999 | 100% |
| noisy | 0.787 | 53% | 0.999 | 100% |
| tilted | 0.804 | 67% | 0.998 | 100% |
| shadow | 0.784 | 50% | 0.999 | 100% |
| clutter | 0.433 | 0% | 0.998 | 100% |
| hard | 0.476 | 3% | 0.997 | 100% |
