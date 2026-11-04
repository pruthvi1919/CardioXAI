Data: REAL MIT-BIH DS2 signals and labels; images are SYNTHETIC renderings (not photos) | 528 segments of 4 s | digitizer failures: 0
~half of the segments contain S/V/F beats; compare rows with each other, not with the full-test 0.59

| Condition | Det. sens | Det. PPV | Macro-F1 (N,S,V) | V sens | V PPV | S sens | PVC burden true -> pred % | HR MAE bpm |
|---|---|---|---|---|---|---|---|---|
| raw_gt | 0.998 | 1.000 | 0.716 | 0.939 | 0.841 | 0.187 | 6.8 -> 7.6 | 0.000 |
| raw_detector | 0.998 | 0.922 | 0.612 | 0.939 | 0.517 | 0.195 | 6.8 -> 11.4 | 6.464 |
| oracle/clean | 0.999 | 0.850 | 0.439 | 0.980 | 0.218 | 0.024 | 6.8 -> 26.0 | 14.193 |
| oracle/realworld | 0.998 | 0.853 | 0.415 | 0.918 | 0.175 | 0.041 | 6.8 -> 30.5 | 13.548 |

Class support (true beats): N=1889, S=123, V=147, F=9
