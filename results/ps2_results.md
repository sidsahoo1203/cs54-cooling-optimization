# CS54 Sub-problem (b) — learned thermal dynamics (v2)

Month `20-06` · 5min bins · state `ambient` [degC] · action `fan_speed` [RPM] · seed 42

Target is the **change** in rack inlet temperature, not its level. MAPE is omitted because the true change passes through zero.

## Dynamics vs step size, with spatial ablation

|   horizon_min | feature_set     |   n_features |     RMSE |      MAE |           R2 |     n |
|--------------:|:----------------|-------------:|---------:|---------:|-------------:|------:|
|             5 | persistence (0) |            0 | 0.174607 | 0.10292  | -4.16521e-06 | 82957 |
|             5 | local_only      |           50 | 0.170632 | 0.102538 |  0.0450138   | 82957 |
|             5 | plus_neighbours |           56 | 0.169653 | 0.102006 |  0.0559442   | 82957 |
|             5 | plus_row        |           59 | 0.171045 | 0.104401 |  0.0403853   | 82957 |
|            15 | persistence (0) |            0 | 0.32973  | 0.189742 | -8.36959e-06 | 82859 |
|            15 | local_only      |           50 | 0.323754 | 0.191323 |  0.0359122   | 82859 |
|            15 | plus_neighbours |           56 | 0.322206 | 0.189073 |  0.0451062   | 82859 |
|            15 | plus_row        |           59 | 0.32373  | 0.192765 |  0.0360502   | 82859 |
|            30 | persistence (0) |            0 | 0.44709  | 0.260008 | -1.54924e-05 | 82712 |
|            30 | local_only      |           50 | 0.434369 | 0.260008 |  0.05608     | 82712 |
|            30 | plus_neighbours |           56 | 0.430929 | 0.259061 |  0.0709703   | 82712 |
|            30 | plus_row        |           59 | 0.439055 | 0.27239  |  0.0356023   | 82712 |
|            60 | persistence (0) |            0 | 0.578603 | 0.342938 | -4.56139e-05 | 82418 |
|            60 | local_only      |           50 | 0.55726  | 0.346038 |  0.072373    | 82418 |
|            60 | plus_neighbours |           56 | 0.557052 | 0.351814 |  0.0730629   | 82418 |
|            60 | plus_row        |           59 | 0.553392 | 0.353921 |  0.0852053   | 82418 |

## Rollout against a persistence rollout

|   horizon_min |   twin_RMSE |   persist_RMSE |   twin_R2 |   persist_R2 |   improvement_% |    n |
|--------------:|------------:|---------------:|----------:|-------------:|----------------:|-----:|
|             5 |   0.0840202 |      0.0832038 |  0.997412 |     0.997462 |       -0.981169 | 1120 |
|            15 |   0.164722  |      0.164105  |  0.990011 |     0.990085 |       -0.376206 | 1120 |
|            30 |   0.278053  |      0.274888  |  0.971879 |     0.972515 |       -1.1514   | 1120 |
|            60 |   0.294401  |      0.288242  |  0.968435 |     0.969742 |       -2.13678  | 1120 |

## Sensitivity to `fan_speed`

|   fan_speed |   mean_predicted_delta_degC |
|------------:|----------------------------:|
|     4299    |                  0.0091432  |
|     4395.35 |                  0.00691922 |
|     4491.71 |                 -0.0025109  |
|     4588.06 |                 -0.0026343  |
|     4684.42 |                 -0.0026343  |
|     4780.77 |                 -0.0026343  |
|     4877.12 |                 -0.00379671 |
|     4973.48 |                 -0.00379671 |
|     5069.83 |                 -0.0122309  |

Node fans are firmware-controlled and react to temperature; this is correlation, not control.

## Thermal envelope

- Standard: ASHRAE TC 9.9 Thermal Guidelines, 4th ed. (2015); Class A1
- Measure: dry-bulb at the equipment air inlet
- Column `ambient_max`: hottest node inlet reading per rack-bin (IPMI `ambient` is the node BMC inlet-side air sensor)
- n = 414,866 rack-bins; range 12.40-40.00 degC; mean 22.89 degC
- within recommended 18.0-27.0 degC: 80.97%
- within allowable A1 15.0-32.0 degC: 98.03%
- below 18.0 degC (over-cooling headroom): 9.73%
