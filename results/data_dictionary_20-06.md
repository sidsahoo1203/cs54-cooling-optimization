# CS54 — M100 data dictionary (20-06)

Siddharth Sahoo · RA2512005010041

## Inventory present on disk

### `ganglia_pub` — 33 metrics

`boottime`, `bytes_in`, `bytes_out`, `cpu_aidle`, `cpu_idle`, `cpu_nice`, `cpu_num`, `cpu_speed`, `cpu_steal`, `cpu_system`, `cpu_user`, `cpu_wio`, `disk_free`, `disk_total`, `gexec`, `load_fifteen`, `load_five`, `load_one`, `machine_type`, `mem_buffers`, `mem_cached`, `mem_free`, `mem_shared`, `mem_total`, `os_name`, `os_release`, `part_max_used`, `pkts_in`, `pkts_out`, `proc_run`, `proc_total`, `swap_free`, `swap_total`

### `ipmi_pub` — 104 metrics

`ambient`, `dimm0_temp`, `dimm10_temp`, `dimm11_temp`, `dimm12_temp`, `dimm13_temp`, `dimm14_temp`, `dimm15_temp`, `dimm1_temp`, `dimm2_temp`, `dimm3_temp`, `dimm4_temp`, `dimm5_temp`, `dimm6_temp`, `dimm7_temp`, `dimm8_temp`, `dimm9_temp`, `fan0_0`, `fan0_1`, `fan1_0`, `fan1_1`, `fan2_0`, `fan2_1`, `fan3_0`, `fan3_1`, `fan_disk_power`, `gpu0_core_temp`, `gpu0_mem_temp`, `gpu1_core_temp`, `gpu1_mem_temp`, `gpu3_core_temp`, `gpu3_mem_temp`, `gpu4_core_temp`, `gpu4_mem_temp`, `gv100card0`, `gv100card1`, `gv100card3`, `gv100card4`, `p0_core0_temp`, `p0_core10_temp`, `p0_core11_temp`, `p0_core12_temp`, `p0_core13_temp`, `p0_core14_temp`, `p0_core15_temp`, `p0_core16_temp`, `p0_core17_temp`, `p0_core18_temp`, `p0_core19_temp`, `p0_core1_temp`, `p0_core20_temp`, `p0_core21_temp`, `p0_core22_temp`, `p0_core23_temp`, `p0_core2_temp`, `p0_core3_temp`, `p0_core4_temp`, `p0_core5_temp`, `p0_core6_temp`, `p0_core7_temp`, `p0_core8_temp`, `p0_core9_temp`, `p0_io_power`, `p0_mem_power`, `p0_power`, `p0_vdd_temp`, `p1_core0_temp`, `p1_core10_temp`, `p1_core11_temp`, `p1_core12_temp`, `p1_core13_temp`, `p1_core14_temp`, `p1_core15_temp`, `p1_core16_temp`, `p1_core17_temp`, `p1_core18_temp`, `p1_core19_temp`, `p1_core1_temp`, `p1_core20_temp`, `p1_core21_temp`, `p1_core22_temp`, `p1_core23_temp`, `p1_core2_temp`, `p1_core3_temp`, `p1_core4_temp`, `p1_core5_temp`, `p1_core6_temp`, `p1_core7_temp`, `p1_core8_temp`, `p1_core9_temp`, `p1_io_power`, `p1_mem_power`, `p1_power`, `p1_vdd_temp`, `pcie`, `ps0_input_power`, `ps0_input_voltag`, `ps0_output_curre`, `ps0_output_volta`, `ps1_input_power`, `ps1_input_voltag`, `ps1_output_curre`, `ps1_output_volta`, `total_power`

### `job_table` — 1 metrics

`job_info_marconi100`

### `nagios_pub` — 1 metrics

`state`

### `slurm_pub` — 20 metrics

`cluster_cpu_util`, `cluster_memory_util`, `job_id`, `num_nodes`, `total_cpus_alloc`, `total_cpus_config`, `total_cpus_down`, `total_cpus_eligible`, `total_cpus_idle`, `total_memory_alloc`, `total_memory_config`, `total_memory_down`, `total_memory_eligible`, `total_memory_idle`, `total_nodes_alloc`, `total_nodes_config`, `total_nodes_down`, `total_nodes_eligible`, `total_nodes_idle`, `total_nodes_mixed`

## Metrics selected for this study

| plugin      | metric              |   n_files |   size_mb | columns                | time_col   | value_col   | entity_col   |   entities_in_sample | sample_time_min     | sample_time_max     |        val_min |         val_max |         val_mean |   val_null_% |
|:------------|:--------------------|----------:|----------:|:-----------------------|:-----------|:------------|:-------------|---------------------:|:--------------------|:--------------------|---------------:|----------------:|-----------------:|-------------:|
| ipmi_pub    | total_power         |         1 |      67.6 | timestamp, value, node | timestamp  | value       | node         |                    2 | 2020-06-06 22:00:40 | 2020-06-07 21:59:40 |  380           |  1300           |    729.488       |            0 |
| ipmi_pub    | ambient             |         1 |      59.2 | timestamp, value, node | timestamp  | value       | node         |                    2 | 2020-06-20 22:00:20 | 2020-06-21 21:59:40 |   12.6         |    15.2         |     13.4215      |            0 |
| ipmi_pub    | p0_power            |         1 |      75.8 | timestamp, value, node | timestamp  | value       | node         |                    2 | 2020-06-23 22:00:00 | 2020-06-24 21:59:40 |   22           |   194           |     62.5316      |            0 |
| ipmi_pub    | p1_power            |         1 |      77.8 | timestamp, value, node | timestamp  | value       | node         |                    3 | 2020-06-10 22:00:00 | 2020-06-11 21:59:40 |   12           |    92           |     29.1104      |            0 |
| ipmi_pub    | fan0_0              |         1 |      42.6 | timestamp, value, node | timestamp  | value       | node         |                    2 | 2020-06-16 22:00:00 | 2020-06-17 21:59:40 | 4300           |  4400           |   4346.4         |            0 |
| ipmi_pub    | fan0_1              |         1 |      28.8 | timestamp, value, node | timestamp  | value       | node         |                   15 | 2020-06-30 22:00:00 | 2020-06-30 23:59:40 |    0           |  6000           |   4713.76        |            0 |
| ipmi_pub    | p0_core0_temp       |         1 |      28.3 | timestamp, value, node | timestamp  | value       | node         |                    2 | 2020-06-17 22:00:00 | 2020-06-18 21:59:40 |   28           |    59           |     37.0928      |            0 |
| ipmi_pub    | p1_core0_temp       |         1 |      25.8 | timestamp, value, node | timestamp  | value       | node         |                    2 | 2020-06-20 22:00:00 | 2020-06-21 21:59:40 |   31           |    47           |     37.6872      |            0 |
| ganglia_pub | cpu_user            |         1 |      24.1 | timestamp, value, node | timestamp  | value       | node         |                    9 | 2020-06-07 22:00:32 | 2020-06-08 14:19:20 |    0           |   100           |     31.0092      |            0 |
| ganglia_pub | cpu_system          |         1 |      22.4 | timestamp, value, node | timestamp  | value       | node         |                    6 | 2020-06-02 22:00:03 | 2020-06-03 21:59:58 |    0           |     1.6         |      0.4605      |            0 |
| ganglia_pub | cpu_idle            |         1 |      23.2 | timestamp, value, node | timestamp  | value       | node         |                    6 | 2020-06-04 22:00:58 | 2020-06-05 21:59:54 |   52.4         |   100           |     89.4303      |            0 |
| ganglia_pub | cpu_wio             |         1 |      20.1 | timestamp, value, node | timestamp  | value       | node         |                    6 | 2020-06-02 22:00:36 | 2020-06-25 21:59:56 |    0           |     0.1         |      0           |            0 |
| ganglia_pub | load_one            |         1 |      34   | timestamp, value, node | timestamp  | value       | node         |                    4 | 2020-06-27 22:00:09 | 2020-06-28 21:59:00 |    0           |    64.68        |     29.5961      |            0 |
| ganglia_pub | load_five           |         1 |      32.8 | timestamp, value, node | timestamp  | value       | node         |                    3 | 2020-06-18 22:00:16 | 2020-06-19 21:59:57 |    0           |    21.95        |      2.3865      |            0 |
| ganglia_pub | mem_free            |         1 |      72.3 | timestamp, value, node | timestamp  | value       | node         |                    6 | 2020-06-05 22:00:22 | 2020-06-13 21:59:21 |    1.93083e+08 |     3.18631e+08 |      2.87536e+08 |            0 |
| ganglia_pub | bytes_in            |         1 |      30.2 | timestamp, value, node | timestamp  | value       | node         |                    6 | 2020-06-03 22:01:14 | 2020-06-23 21:59:59 | 4737.94        |     2.25368e+08 | 369725           |            0 |
| ganglia_pub | bytes_out           |         1 |      30.9 | timestamp, value, node | timestamp  | value       | node         |                   12 | 2020-06-01 22:00:12 | 2020-06-02 21:59:40 | 2165.78        | 70845.8         |  10801           |            0 |
| ganglia_pub | proc_run            |         1 |      22.3 | timestamp, value, node | timestamp  | value       | node         |                    9 | 2020-06-17 22:00:48 | 2020-06-18 14:10:25 |    0           |   133           |     13.471       |            0 |
| slurm_pub   | cluster_cpu_util    |         1 |       1   | timestamp, value       | timestamp  | value       |              |                    0 | 2020-06-04 23:00:00 | 2020-06-30 22:00:00 |   16           |    75           |     48.5212      |            0 |
| slurm_pub   | cluster_memory_util |         1 |       1   | timestamp, value       | timestamp  | value       |              |                    0 | 2020-06-04 13:53:25 | 2020-06-30 22:00:00 |   17           |    72           |     48.33        |            0 |
