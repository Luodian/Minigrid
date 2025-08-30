# Repository Guidelines

## Project Structure & Module Organization
- `minigrid/`: core package. Environments in `minigrid/envs/` (incl. `babyai/` and `wfc/`), wrappers in `minigrid/wrappers.py`, utilities in `minigrid/utils/`.
- `tests/`: pytest suite (e.g., `tests/test_envs.py`, `tests/test_wrappers.py`, `tests/test_wfc/`).
- `docs/`: Sphinx site and build scripts in `docs/_scripts/`.
- `figures/`: images and media used in docs.
- Scripts: `minimal_example.py`, `train_rl_agent.py`, `record_video.py`, `record_trajectory.py`, `train_all_envs.sh`.

## Build, Test, and Development Commands
- Setup (recommended): `python -m venv .venv && source .venv/bin/activate`.
- Install: `pip install -e .[testing,wfc]` (adds pytest and WFC extras). For docs: `pip install -r docs/requirements.txt`.
- Lint/format: `pre-commit install && pre-commit run --all-files`.
- Tests: `pytest` and `pytest --doctest-modules minigrid/`.
- Docs: `make -C docs html` (outputs to `docs/_build/` or `_build/`).
- Run an env manually: `python minigrid/manual_control.py --env MiniGrid-Empty-8x8-v0`.

## Coding Style & Naming Conventions
- Formatting: Black; imports: isort (Black profile); linting: flake8; spelling: codespell. All wired via pre-commit.
- Indentation: 4 spaces; max line length: follow Black (flake8 configured accordingly).
- Types: use Python type hints; prefer `from __future__ import annotations` (added by isort config).
- Naming: modules and functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.

## Testing Guidelines
- Framework: pytest. Place tests under `tests/`, name files `test_*.py` and functions `test_*`.
- Add unit tests for new envs, wrappers, and utilities; keep doctests passing (`pytest --doctest-modules minigrid/`).
- WFC tests require the `[wfc]` extra.
- Run a subset: `pytest tests/test_envs.py -k LavaCrossing`.

## Commit & Pull Request Guidelines
- Commits: imperative and scoped (e.g., "Add ObstructedMaze v1 wrapper"); reference issues (`Fixes #123`).
- PRs: use the template, provide a clear description, link issues, include screenshots/gifs for visual changes, update docs, add/adjust tests, and run `pre-commit`.
- CI runs `pytest` and doctests across 3.8–3.12; ensure compatibility.

## Environment & Compatibility
- Supported: Python 3.8–3.12 on Linux/macOS. Follow the Gymnasium API and avoid breaking env registration or wrapper behavior.
