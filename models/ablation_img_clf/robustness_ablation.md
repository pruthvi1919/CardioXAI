Data: REAL MIT-BIH DS2 signals and labels; images are SYNTHETIC renderings (not photos) | 258 segments of 8 s | digitizer failures: 0
~half of the segments contain S/V/F beats; compare rows with each other, not with the full-test 0.59

| Condition | Det. sens | Det. PPV | Macro-F1 (N,S,V) | V sens | V PPV | S sens | PVC burden true -> pred % | HR MAE bpm |
|---|---|---|---|---|---|---|---|---|
| raw_gt | 1.000 | 1.000 | 0.639 | 0.525 | 0.970 | 0.642 | 5.2 -> 2.8 | 0.000 |
| raw_detector | 0.999 | 0.986 | 0.603 | 0.516 | 0.708 | 0.617 | 5.2 -> 3.8 | 1.332 |
| oracle/clean | 0.995 | 0.960 | 0.630 | 0.754 | 0.424 | 0.567 | 5.2 -> 9.0 | 3.613 |
| unet/clean | 0.995 | 0.907 | 0.542 | 0.844 | 0.256 | 0.425 | 5.2 -> 15.7 | 7.896 |
| oracle/hard | 0.974 | 0.947 | 0.585 | 0.721 | 0.367 | 0.538 | 5.3 -> 10.1 | 3.709 |
| unet/hard | 0.962 | 0.889 | 0.517 | 0.754 | 0.242 | 0.415 | 5.2 -> 15.1 | 6.985 |
| oracle/realworld | 0.994 | 0.943 | 0.557 | 0.661 | 0.305 | 0.475 | 5.2 -> 10.7 | 4.373 |
| unet/realworld | 0.764 | 0.746 | 0.361 | 0.492 | 0.163 | 0.208 | 5.3 -> 15.6 | 10.811 |

Class support (true beats): N=2083, S=120, V=122, F=15
