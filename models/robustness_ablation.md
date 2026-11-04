Data: REAL MIT-BIH DS2 signals and labels; images are SYNTHETIC renderings (not photos) | 258 segments of 8 s | digitizer failures: 0
~half of the segments contain S/V/F beats; compare rows with each other, not with the full-test 0.59

| Condition | Det. sens | Det. PPV | Macro-F1 (N,S,V) | V sens | V PPV | S sens | PVC burden true -> pred % | HR MAE bpm |
|---|---|---|---|---|---|---|---|---|
| raw_gt | 1.000 | 1.000 | 0.732 | 0.967 | 0.792 | 0.250 | 5.2 -> 6.4 | 0.000 |
| raw_detector | 1.000 | 0.934 | 0.613 | 0.959 | 0.457 | 0.242 | 5.2 -> 10.2 | 5.414 |
| oracle/clean | 1.000 | 0.809 | 0.314 | 0.984 | 0.088 | 0.008 | 5.2 -> 47.1 | 18.132 |
| oracle_ext/clean | 0.999 | 0.840 | 0.334 | 0.025 | 0.125 | 0.070 | 5.2 -> 0.9 | 14.606 |
| unet/clean | 1.000 | 0.718 | 0.233 | 0.992 | 0.055 | 0.000 | 5.2 -> 67.5 | 29.860 |
| unet_ext/clean | 1.000 | 0.799 | 0.233 | 1.000 | 0.059 | 0.042 | 5.2 -> 70.2 | 19.392 |
| oracle/hard | 0.975 | 0.848 | 0.317 | 0.877 | 0.083 | 0.051 | 5.3 -> 48.6 | 12.111 |
| oracle_ext/hard | 0.890 | 0.751 | 0.295 | 0.016 | 0.062 | 0.094 | 5.3 -> 1.2 | 15.285 |
| unet/hard | 0.966 | 0.744 | 0.226 | 0.934 | 0.058 | 0.008 | 5.2 -> 64.9 | 22.699 |
| unet_ext/hard | 0.969 | 0.781 | 0.251 | 0.926 | 0.064 | 0.025 | 5.2 -> 61.7 | 18.538 |
| oracle/realworld | 0.997 | 0.866 | 0.279 | 0.950 | 0.075 | 0.025 | 5.2 -> 57.5 | 11.449 |
| oracle_ext/realworld | 1.000 | 0.833 | 0.355 | 0.041 | 0.227 | 0.167 | 5.2 -> 0.8 | 15.260 |
| unet/realworld | 0.778 | 0.652 | 0.162 | 0.746 | 0.048 | 0.000 | 5.3 -> 69.3 | 23.357 |
| unet_ext/realworld | 0.789 | 0.679 | 0.186 | 0.746 | 0.051 | 0.017 | 5.3 -> 67.0 | 19.885 |

Class support (true beats): N=2083, S=120, V=122, F=15
