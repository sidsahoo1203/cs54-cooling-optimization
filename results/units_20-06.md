# CS54 — units and meanings (20-06)

| column | unit | meaning |
|---|---|---|
| `total_power` | W | node total power draw (= heat dissipated) |
| `ambient` | degC | node ambient/inlet air temperature (BMC sensor) |
| `p0_power` | W | CPU socket 0 power |
| `p1_power` | W | CPU socket 1 power |
| `cpu_power` | W | p0_power + p1_power |
| `fan0_0` | RPM | node fan 0 tachometer |
| `fan0_1` | RPM | node fan 1 tachometer |
| `fan_speed` | RPM | mean of fan0_0 and fan0_1 |
| `p0_core0_temp` | degC | CPU0 core 0 die temperature |
| `p1_core0_temp` | degC | CPU1 core 0 die temperature |
| `core_temp` | degC | mean of p0_core0_temp and p1_core0_temp |
| `cpu_user` | % | CPU time in user mode |
| `cpu_system` | % | CPU time in kernel mode |
| `cpu_idle` | % | CPU time idle |
| `cpu_wio` | % | CPU time waiting on I/O |
| `cpu_busy` | % | 100 - cpu_idle (directly measured complement) |
| `cpu_busy_alt` | % | cpu_user + cpu_system + cpu_wio (cross-check) |
| `load_one` | procs | 1-minute load average |
| `load_five` | procs | 5-minute load average |
| `mem_free` | kB | free memory, KILOBYTES (Ganglia convention) |
| `bytes_in` | B/s | network bytes received per second |
| `bytes_out` | B/s | network bytes sent per second |
| `proc_run` | count | runnable processes |
| `cluster_cpu_util` | % | cluster-wide CPU utilisation (SLURM) |
| `cluster_memory_util` | % | cluster-wide memory utilisation (SLURM) |

All per-node metrics are averaged across the nodes in a rack, so every value is *mean per node in that rack*. Temperature columns additionally carry `_max`, the rack hotspot.

Thermal standard: ASHRAE TC 9.9 Thermal Guidelines, 4th ed. (2015); Class A1. Recommended 18.0-27.0 degC, allowable 15.0-32.0 degC dry-bulb at the equipment inlet.
