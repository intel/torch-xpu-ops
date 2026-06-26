# Run Test — Shared Reference

Used by `fix/reproduce` and `fix/verify`. Covers running tests and
interpreting results. Test path resolution specifics (e.g. submodule paths,
CI metadata format) are in the domain skill loaded by the orchestrator.

## Running tests

Run from `$PYTORCH_DIR/`:
```bash
# pytest
pytest -xvs "<resolved_path>::ClassName::test_method"

# python unittest
python <resolved_path> ClassName.test_method
```

Keep the original format (pytest vs python). Timeout: 10 minutes.

Validate the file exists before running:
```bash
ls -la $PYTORCH_DIR/<resolved_path>
```
If not found → `CANNOT_VERIFY`.

For `-k` filters, validate with `--collect-only`:
```bash
pytest --collect-only -q "<file>" -k "<filter>" 2>&1 | tail -5
```
`0 items collected` → `CANNOT_VERIFY`.

## Result interpretation

### PASSED
- pytest exit 0, `N passed`, no `xfailed`, no `all skipped`
- unittest exit 0, `Ran N tests... OK`, actual tests ran

### FAILED
- pytest exit 1 with test failures
- unittest `FAILED` or `ERROR`
- Any actual test failure output
- All tests `xfailed` (expected failure = bug still present)

### CANNOT_VERIFY
- `collected 0 items` / `no tests ran`
- File not found on disk
- All tests skipped (cannot confirm either way)
- Timeout
