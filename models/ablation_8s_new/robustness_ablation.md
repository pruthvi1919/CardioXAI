Data: REAL MIT-BIH DS2 signals and labels; images are SYNTHETIC renderings (not photos) | 258 segments of 8 s | digitizer failures: 0
~half of the segments contain S/V/F beats; compare rows with each other, not with the full-test 0.59

| Condition | Det. sens | Det. PPV | Macro-F1 (N,S,V) | V sens | V PPV | S sens | PVC burden true -> pred % | HR MAE bpm |
|---|---|---|---|---|---|---|---|---|
| raw_gt | 1.000 | 1.000 | 0.732 | 0.967 | 0.792 | 0.250 | 5.2 -> 6.4 | 0.000 |
| raw_detector | 0.999 | 0.986 | 0.699 | 0.959 | 0.692 | 0.242 | 5.2 -> 7.1 | 1.332 |
| oracle/clean | 0.995 | 0.960 | 0.334 | 0.959 | 0.120 | 0.008 | 5.2 -> 40.3 | 3.613 |
| unet/clean | 0.995 | 0.907 | 0.244 | 0.967 | 0.072 | 0.000 | 5.2 -> 63.8 | 7.896 |
| oracle/hard | 0.974 | 0.947 | 0.329 | 0.893 | 0.105 | 0.043 | 5.3 -> 43.6 | 3.709 |
| unet/hard | 0.962 | 0.889 | 0.243 | 0.934 | 0.074 | 0.008 | 5.2 -> 61.3 | 6.985 |
| oracle/realworld | 0.994 | 0.943 | 0.284 | 0.909 | 0.083 | 0.025 | 5.2 -> 54.4 | 4.373 |
| unet/realworld | 0.764 | 0.746 | 0.167 | 0.705 | 0.054 | 0.000 | 5.3 -> 67.4 | 10.811 |

Class support (true beats): N=2083, S=120, V=122, F=15
