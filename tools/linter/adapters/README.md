# lintrunner adapters

These files adapt our various linters to work with `lintrunner`.

## Layout
- `.lintrunner.toml` (repo root): linter definitions, grouped by language.
- `tools/linter/lintrunner.sh`: CI/local entrypoint; runs one language group (`LINT_GROUP`) in full or incremental scope.
- `tools/linter/adapters/`: one adapter per linter, plus `s3_init.py`/`update_s3.py` for prebuilt clang binaries.
- `.lintbin/` (git-ignored): downloaded binaries, the lint venv and `lint.json`.
- Tool configs stay where the tools discover them: `.clang-format`, `.clang-tidy`, `.cmakelintrc`, `pyproject.toml`, `.github/actionlint.yaml`, `.github/zizmor.yml`, `.pre-commit-config.yaml`.

## Adding a new linter
1. init and linter
2. {{DRYRUN}} and {{PATHSFILE}}
3. never exit uncleanly
4. Communication protocol
5. Self-contained
6. Add its code to a `LINT_GROUPS` entry in `tools/linter/lintrunner.sh` (CI fails otherwise)
