"""Runs one hanging step in its own process, so a test can interrupt it as a person would.

Usage: one_step.py OUT_ROOT MODE[=hang]   -> prints the run directory on the first line
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run.plans import Plan, Step                            # noqa: E402
from run.runner import (Run, RunInterrupted, RunOptions,    # noqa: E402
                        interrupt_exit_code)

root = Path(sys.argv[1])
opts = RunOptions(out_root=root, sample_interval_s=0.2, pre_roll_s=0.1, post_roll_s=0.1,
                  capture_env=False)
run = Run(Plan("fake", "test", [Step("hang", [sys.executable,
          str(Path(__file__).with_name("fake_probe.py")), "hang"], timeout_s=600)]), opts,
          run_id="interrupted")
print(run.dir, flush=True)
try:
    run.execute()
except (KeyboardInterrupt, RunInterrupted) as e:            # what `./echolock run` does
    print(f"interrupted by {run.interrupted}", file=sys.stderr, flush=True)
    sys.exit(interrupt_exit_code(e))
