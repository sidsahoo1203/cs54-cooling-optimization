# CS54 Sub-problem (b) — learned thermal dynamics (v2)

Month `20-06` · 5min bins · state `ambient` [degC] · action `fan_speed` [RPM] · seed 42

Target is the **change** in rack inlet temperature, not its level. MAPE is omitted because the true change passes through zero.

## Dynamics vs step size, with spatial ablation

|   horizon_min | feature_set     |   n_features |     RMSE |      MAE |           R2 |    n |
|--------------:|:----------------|-------------:|---------:|---------:|-------------:|-----:|
|             5 | persistence (0) |            0 | 0.186788 | 0.119689 | -2.04713e-05 | 1546 |
|             5 | local_only      |           50 | 0.186787 | 0.119705 | -8.59175e-06 | 1546 |
|             5 | plus_neighbours |           56 | 0.185805 | 0.120284 |  0.0104709   | 1546 |
|             5 | plus_row        |           59 | 0.186381 | 0.119772 |  0.00433307  | 1546 |
|            15 | persistence (0) |            0 | 0.298263 | 0.202224 | -9.30773e-05 | 1544 |
|            15 | local_only      |           50 | 0.293703 | 0.199775 |  0.0302577   | 1544 |
|            15 | plus_neighbours |           56 | 0.29407  | 0.199742 |  0.0278312   | 1544 |
|            15 | plus_row        |           59 | 0.294525 | 0.199887 |  0.0248174   | 1544 |
|            30 | persistence (0) |            0 | 0.355546 | 0.257671 | -0.000140477 | 1541 |
|            30 | local_only      |           50 | 0.344579 | 0.25206  |  0.0606125   | 1541 |
|            30 | plus_neighbours |           56 | 0.344072 | 0.250382 |  0.0633741   | 1541 |
|            30 | plus_row        |           59 | 0.345149 | 0.25255  |  0.0574994   | 1541 |
|            60 | persistence (0) |            0 | 0.378826 | 0.285155 | -0.000228552 | 1535 |
|            60 | local_only      |           50 | 0.377227 | 0.283737 |  0.00820021  | 1535 |
|            60 | plus_neighbours |           56 | 0.375243 | 0.282006 |  0.018606    | 1535 |
|            60 | plus_row        |           59 | 0.375161 | 0.281548 |  0.0190348   | 1535 |

## Rollout against a persistence rollout

|   horizon_min |   twin_RMSE |   persist_RMSE |   twin_R2 |   persist_R2 |   improvement_% |   n |
|--------------:|------------:|---------------:|----------:|-------------:|----------------:|----:|
|             5 |    0.163246 |       0.164095 |  0.871301 |     0.869959 |        0.517394 | 128 |
|            15 |    0.254153 |       0.255326 |  0.692641 |     0.689798 |        0.459335 | 128 |
|            30 |    0.330794 |       0.34646  |  0.505047 |     0.457055 |        4.5218   | 128 |
|            60 |    0.346587 |       0.347875 |  0.411842 |     0.407462 |        0.370304 | 127 |

## Sensitivity to `fan_speed`

|   fan_speed |   mean_predicted_delta_degC |
|------------:|----------------------------:|
|     4485.67 |                  0.00802968 |
|     4495.23 |                  0.00802968 |
|     4504.79 |                  0.00802968 |
|     4514.35 |                  0.00802968 |
|     4523.9  |                  0.00802968 |
|     4533.46 |                  0.00803672 |
|     4543.02 |                  0.00803672 |
|     4552.58 |                  0.00803672 |
|     4562.14 |                  0.00803672 |

Node fans are firmware-controlled and react to temperature; this is correlation, not control.

## Thermal envelope

- Standard: ASHRAE TC 9.9 Thermal Guidelines, 4th ed. (2015); Class A1
- Measure: dry-bulb at the equipment air inlet
- Column `ambient_max`: hottest node inlet reading per rack-bin (IPMI `ambient` is the node BMC inlet-side air sensor)
- n = 414,866 rack-bins; range 12.40-40.00 degC; mean 22.89 degC
- within recommended 18.0-27.0 degC: 80.97%
- within allowable A1 15.0-32.0 degC: 98.03%
- below 18.0 degC (over-cooling headroom): 9.73%
