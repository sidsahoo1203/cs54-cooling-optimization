# CS54 Sub-problem (c) — controller results

Month `22-01` · 5-min control interval · forecast horizon 15 min · violation weight 50.0 · 3 seeds · 100,000 timesteps

| policy                        |   cooling_kWh_mean |   cooling_kWh_std |   violation_rate |   violation_degC_steps |   reward |   episodes | mode     |   n_seeds |
|:------------------------------|-------------------:|------------------:|-----------------:|-----------------------:|---------:|-----------:|:---------|----------:|
| fixed_free_cooling_valve_6.6% |            4199.77 |          189.867  |                0 |                      0 | -283.791 |         10 | nan      |       nan |
| reactive_rule                 |            4209.14 |          182.791  |                0 |                      0 | -284.424 |         10 | nan      |       nan |
| ppo_none                      |            4197.58 |           11.3677 |                0 |                      0 | -283.643 |         30 | none     |         3 |
| ppo_oracle                    |            4189.16 |           21.3871 |                0 |                      0 | -283.073 |         30 | oracle   |         3 |
| ppo_forecast                  |            4178.46 |           29.3347 |                0 |                      0 | -282.35  |         30 | forecast |         3 |

## Decomposition

- value of anticipation (none -> oracle): **+0.20%**
- cost of imperfect prediction (oracle -> real): **+0.26%**
