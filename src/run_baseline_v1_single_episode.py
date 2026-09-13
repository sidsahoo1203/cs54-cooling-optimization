"""
Phase 1 baseline: fixed-setpoint rule-based cooling controller.
Runs one episode on SustainDC and records per-timestep metrics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sustaindc'))

import pandas as pd
import random
import numpy as np
from sustaindc_env import SustainDC, EnvConfig

def run_episode(seed=42):
    config = dict(EnvConfig.DEFAULT_CONFIG)
    config['agents'] = ['agent_dc']
    config['flexible_load'] = 0.0
    config['month'] = 0  # 0-indexed: 0 = January

    env = SustainDC(config)
    random.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    info = env.infos

    records = []
    done = False
    step = 0

    while not done and step < 1000:
        # Action 1 = hold setpoint (rule-based baseline)
        actions = {'agent_dc': 1}
        obs, reward, terminated, truncated, info = env.step(actions)

        row = {'step': step}
        for k, v in info.items():
            if isinstance(v, dict):
                row.update(v)
            else:
                row[k] = v
        records.append(row)

        done = terminated.get('__all__', False) or truncated.get('__all__', False)
        step += 1

    return pd.DataFrame(records)

if __name__ == '__main__':
    df = run_episode()
    print(f"Steps completed: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    df.to_csv('results/baseline_raw.csv', index=False)
    print("Saved to results/baseline_raw.csv")
