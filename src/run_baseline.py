"""
Phase 1: Rule-based fixed-setpoint baseline for CS54.
Runs SustainDC across multiple seeds, records per-timestep metrics
including rack inlet temperatures, and computes PUE, cooling energy,
and thermal safety margins.
"""
import sys, os, json, random, subprocess
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sustaindc'))

import numpy as np
import pandas as pd
from sustaindc_env import SustainDC, EnvConfig

SEEDS = [42, 123, 456, 789, 1011]
ASHRAE_MIN = 18.0
ASHRAE_MAX = 27.0
HOLD_ACTION = 1          # 1 = maintain setpoint
STEPS_PER_HOUR = 4       # 15-minute timesteps


def git_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return 'no_git'


def make_config():
    cfg = dict(EnvConfig.DEFAULT_CONFIG)
    cfg['agents'] = ['agent_dc']
    cfg['flexible_load'] = 0.0
    cfg['month'] = 0
    return cfg


def run_episode(seed, max_steps=1000):
    """Run one episode holding a fixed setpoint."""
    random.seed(seed)
    np.random.seed(seed)

    env = SustainDC(make_config())
    env.reset()

    rows = []
    for step in range(max_steps):
        _, _, terminated, truncated, info = env.step({'agent_dc': HOLD_ACTION})

        dc = info['agent_dc']
        inlet = list(env.dc_env.dc.rackwise_inlet_temp)

        rows.append({
            'seed': seed,
            'step': step,
            'setpoint': dc['dc_crac_setpoint'],
            'workload': dc['dc_cpu_workload_fraction'],
            'it_power_kW': dc['dc_ITE_total_power_kW'],
            'hvac_power_kW': dc['dc_HVAC_total_power_kW'],
            'total_power_kW': dc['dc_total_power_kW'],
            'outlet_temp_C': dc['dc_int_temperature'],
            'ambient_temp_C': dc['dc_exterior_ambient_temp'],
            'inlet_min_C': min(inlet),
            'inlet_max_C': max(inlet),
            'inlet_mean_C': sum(inlet) / len(inlet),
        })

        if terminated.get('__all__', False) or truncated.get('__all__', False):
            break

    return pd.DataFrame(rows)


def compute_metrics(df):
    """Evaluation harness: energy, PUE, thermal safety."""
    hours_per_step = 1.0 / STEPS_PER_HOUR

    cooling_kWh = df.hvac_power_kW.sum() * hours_per_step
    it_kWh = df.it_power_kW.sum() * hours_per_step
    total_kWh = df.total_power_kW.sum() * hours_per_step

    violations_high = int((df.inlet_max_C > ASHRAE_MAX).sum())
    violations_low = int((df.inlet_min_C < ASHRAE_MIN).sum())

    return {
        'steps': len(df),
        'cooling_energy_kWh': round(cooling_kWh, 2),
        'it_energy_kWh': round(it_kWh, 2),
        'total_energy_kWh': round(total_kWh, 2),
        'pue': round(total_kWh / it_kWh, 4),
        'mean_setpoint_C': round(df.setpoint.mean(), 2),
        'ashrae_violations_high': violations_high,
        'ashrae_violations_low': violations_low,
        'max_inlet_temp_C': round(df.inlet_max_C.max(), 2),
        'mean_headroom_C': round((ASHRAE_MAX - df.inlet_max_C).mean(), 2),
        'min_headroom_C': round((ASHRAE_MAX - df.inlet_max_C).min(), 2),
    }


def main():
    os.makedirs('results', exist_ok=True)

    all_steps = []
    per_seed = []

    for seed in SEEDS:
        print(f'Running seed {seed}...')
        df = run_episode(seed)
        all_steps.append(df)
        m = compute_metrics(df)
        m['seed'] = seed
        per_seed.append(m)
        print(f"  PUE={m['pue']}  cooling={m['cooling_energy_kWh']} kWh  "
              f"max_inlet={m['max_inlet_temp_C']}C")

    steps_df = pd.concat(all_steps, ignore_index=True)
    seeds_df = pd.DataFrame(per_seed)

    steps_df.to_csv('results/baseline_timeseries.csv', index=False)
    seeds_df.to_csv('results/baseline_per_seed.csv', index=False)

    numeric = seeds_df.drop(columns=['seed'])
    summary = {
        'controller': 'rule_based_fixed_setpoint',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_hash(),
        'config': {
            'seeds': SEEDS,
            'action': 'hold (1)',
            'workload_file': make_config()['workload_file'],
            'weather_file': make_config()['weather_file'],
            'month': 0,
            'ashrae_range_C': [ASHRAE_MIN, ASHRAE_MAX],
        },
        'mean': numeric.mean().round(4).to_dict(),
        'std': numeric.std().round(4).to_dict(),
    }

    with open('results/baseline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n--- BASELINE RESULTS (mean +/- std over 5 seeds) ---')
    for k in ['pue', 'cooling_energy_kWh', 'total_energy_kWh',
              'max_inlet_temp_C', 'mean_headroom_C',
              'ashrae_violations_high', 'ashrae_violations_low']:
        print(f'{k:26s} {numeric[k].mean():10.4f}  +/- {numeric[k].std():.4f}')
    print('\nSaved: baseline_timeseries.csv, baseline_per_seed.csv, '
          'baseline_summary.json')


if __name__ == '__main__':
    main()
