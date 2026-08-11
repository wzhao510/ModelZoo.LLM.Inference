# Automated Test Runner

## Entry point

Use the root-level shell script:

```bash
chmod +x ./run_all_tests.sh
./run_all_tests.sh
```

## What it does

- Runs all test files in `test_manifest.txt` sequentially.
- Writes one log directory per run under `artifacts/test_runs/<timestamp>/`.
- Records per-test `PASS` or `FAIL`, duration, raw pytest log, and JUnit XML.
- Continues after failures by default so you can see the full batch result.

## Common usage

```bash
./run_all_tests.sh --list
./run_all_tests.sh --only nixl
./run_all_tests.sh --stop-on-fail
./run_all_tests.sh --python-bin /path/to/python3
```

## Extend the suite

1. Add a new test file under the appropriate subdirectory.
2. Append its relative path to `test_manifest.txt`.
3. Re-run `./run_all_tests.sh`.

If `test_manifest.txt` is removed, the script falls back to auto-discovering `test_*.py`.

## Shared infrastructure

`common.py` (root of `auto_pd_test/`) holds the process lifecycle, port-waiting,
log-dumping, and TTFT/correctness-assertion helpers every test file uses — import
from it instead of re-deriving `_wait_for_port` / `_dump_file` / process cleanup in
a new test file. `conftest.py` registers pytest markers (`pd`, `kvoffload`,
`kvshare`, `backend`, `slow`) and makes `auto_pd_test/` importable from any
subdirectory test.

See `TEST_PLAN.md` for the current test inventory, the framework issues found in
the last audit (and which are already fixed vs. still open), and the phased plan
for expanding coverage.