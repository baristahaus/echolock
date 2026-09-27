# load

Background load generators, the grid's second axis (milestone 4). None are implemented yet.
Each one will run as its own process that the grid runner starts before a probe run and stops
after it. It will record its intended level and its own achieved level in the same JSONL format,
so "50% SM share" can be checked instead of assumed.

Planned:

- SM share: a persistent kernel occupying N% of SMs.
- Bandwidth share: a GPU streaming reader throttled to N% of the measured idle bandwidth.
- CPU memory traffic: host threads streaming at a set rate.
- Unified-memory fill: allocate and touch to 50/80/95% of physical memory, then hold.
- Thermal soak: sustained full load for minutes before and during the probe.
