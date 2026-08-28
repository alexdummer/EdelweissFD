# Contributing to EdelweissFD

We welcome pull requests of all sizes, bug reports, and enhancements. This guide explains how to set up your development environment, follow our commit conventions, run tests, write documentation, and open high-quality issues and pull requests.

---

## Code of Conduct

By participating in this project, you agree to uphold a respectful, inclusive environment. Be kind, be constructive, and assume good intent. If you encounter any problems, please open an issue.

---

## Ways to Contribute

- Report bugs and propose enhancements via GitHub Issues.
- Improve documentation, docstrings, and Python example scripts (`examples/`).
- Add pytest test cases, improve coverage, or benchmark stencils.
- Implement new finite difference stencils (cell-based or GFDM), differential operators, or grid generators.
- Optimize performance-critical routines or compiled Cython kernel fallbacks.

> **Pull requests are welcome!** See [Pull Requests](#pull-requests).

---

## Environment Setup & Installation

EdelweissFD requires **Python >= 3.14** and targets the free-threaded ("nogil") CPython build (`python-freethreading`).

### Prerequisites
[EdelweissFE](https://github.com/EdelweissFE/EdelweissFE) (including its compiled Marmot material interfaces, if Marmot support is needed) must already be installed.

### Installation
```bash
# Editable install
pip install -e .

# With optional testing and documentation dependencies:
pip install -e ".[test,doc]"
```

---

## Pre-commit Hooks

We use **pre-commit** to enforce code formatting and static checks before each commit.

### Install and Run
```bash
# Install the pre-commit framework
pip install pre-commit

# Install git hook scripts
pre-commit install

# Run checks against all files
pre-commit run --all-files
```

Formatting and linting tools configured in `.pre-commit-config.yaml`:
- `autoflake`: Removes unused imports and variables (`--remove-all-unused-imports --in-place --recursive`).
- `black`: Formats Python code (`--line-length 120`).
- `isort`: Sorts imports (`--profile black`).
- `flake8`: Style and lint checks (`--max-line-length 120`, `--ignore E203,E501`, `--extend-ignore W503`).
- `check-yaml` & `trailing-whitespace`: Validates YAML files and removes trailing spaces.

---

## Conventional Commits

All commits **must** follow the [Conventional Commits](https://www.conventionalcommits.org) specification. This maintains a clean Git history and facilitates automated changelog generation.

### Format
```
<type>(<scope>): <description>
```

### Types
- **feat**: A new feature or capability
- **fix**: A bug fix
- **docs**: Documentation-only changes
- **test**: Adding, updating, or fixing tests
- **refactor**: Code changes that neither fix a bug nor add a feature
- **perf**: Performance improvements
- **build**: Build system, dependencies, or packaging
- **ci**: CI workflows and configuration
- **chore**: Routine maintenance tasks

### Scopes
`stencils`, `operators`, `grids`, `drivers`, `materials`, `stepactions`, `tests`, `docs`

### Examples
```
feat(stencils): add Generalized Finite Difference (GFDM) gradient plasticity stencil
fix(operators): correct cell corner gradient operator weights in 3D
perf(stencils): optimize compiled kernel memoryview access
docs(drivers): document indirect arc-length control on FDStep
test(tangents): verify numerical vs analytic tangents for gradient enhanced stencils
```

> Keep the commit subject line concise (≤72 characters). Use the body to explain **what** and **why**, referencing issues where appropriate (e.g., `Fixes #42`).

---

## Opening Issues

Before opening a new issue, search existing issues and pull requests to avoid duplicates. When filing a bug report, please provide:
- **Environment**: OS, Python version (standard vs free-threaded build), and EdelweissFE/Marmot versions/commits.
- **Minimal Reproducible Example (MRE)**: A self-contained Python script (`run.py`) using `FDSimulation` demonstrating the failure.
- **Expected vs Actual Behavior**: Expected results vs observed error tracebacks or discrepancies.

---

## Pull Requests

We follow the GitHub flow: **fork → branch → PR → review → merge**.

### Target Branches
- **Bug fixes (`fix`)**: Open pull requests targeting the `master` branch.
- **Features, enhancements, and refactoring (`feat`, `refactor`, `perf`, etc.)**: Open pull requests targeting the upcoming release branch: `next_v<YY>.<MM>` (e.g., `next_v26.11`).

### Workflow
1. **Fork** the repository and create a feature/bugfix branch from the appropriate target base (`master` for fixes, `next_v<YY>.<MM>` for features):
   ```bash
   # For a bug fix:
   git checkout -b fix/<short-scope>-<concise-topic> origin/master

   # For a new feature / refactoring:
   git checkout -b feat/<short-scope>-<concise-topic> origin/next_v26.11
   ```
2. **Develop & Format**: Make your changes and verify that `pre-commit run --all-files` passes locally.
3. **Test**: Ensure all tests pass (`pytest` and `EDELWEISSFD_NO_COMPILED_KERNEL=1 pytest`).
4. **Open a PR**: Target the correct branch (`master` for bug fixes, `next_v<YY>.<MM>` for features), provide a clear title following Conventional Commits, and describe your changes.

### Synchronizing with EdelweissFE & Marmot
If your changes depend on features or fixes in [EdelweissFE](https://github.com/EdelweissFE/EdelweissFE) or [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/), ensure the sibling branches are named identically. CI automatically resolves and checks out matching branches during test workflows.

### PR Checklist
- [ ] PR targets the correct branch (`master` for bugfixes, `next_v<YY>.<MM>` for features/enhancements).
- [ ] PR title follows Conventional Commits format.
- [ ] `pre-commit run --all-files` passes cleanly.
- [ ] Full test suite passes locally via `pytest` (and `pytest -m "not marmot"`).
- [ ] Dual-path execution verified: `EDELWEISSFD_NO_COMPILED_KERNEL=1 pytest` passes.
- [ ] Stencil tangents verified against central differences in `tests/test_tangents.py`.
- [ ] New features, stencils, or drivers are documented in `doc/source/documentation/`.

---

## Adding & Running Tests

The test suite is driven by **pytest**.

### Running Tests
```bash
# Run complete test suite
pytest

# Run tests that do not require compiled Marmot extensions
pytest -m "not marmot"

# Run a single test file or specific test
pytest tests/test_gradientplasticity.py
pytest tests/test_gradientplasticity.py::test_name

# Force the pure-Python kernel fallback path
EDELWEISSFD_NO_COMPILED_KERNEL=1 pytest

# Run example scripts directly
python examples/TensionBar1D/run.py
python examples/SimpleBeam2D/run.py
```

### Stored Reference Solutions
Reference solutions for reconstructed EdelweissFE benchmarks are stored under `tests/references/`. If expected results change legitimately due to intentional algorithmic updates, regenerate them using:
```bash
python tests/regenerate_references.py
```

---

## Documentation

Documentation is built with **Sphinx** and lives under `doc/source/`.

### Local Documentation Build
```bash
# Build HTML documentation into ./docs
sphinx-build ./doc/source/ ./docs -b html
```

### Documenting New Features
When adding new functionality:
- **Subsystem documentation**: Update the corresponding topic page in `doc/source/documentation/*.rst` (`stencils.rst`, `operators.rst`, `grids.rst`, `drivers.rst`, `materials.rst`, `models.rst`).
- **Docstrings**: Use **NumPy-style docstrings** (`Parameters`, `Returns`, `Raises`, `Notes`) for all public classes, methods, and functions.

---

## Architectural & Coding Guidelines

- **The "Thin Package" Principle**: EdelweissFD adds discretization (grids, stencils, operators, scripting driver) and reuses EdelweissFE's solvers, DOFs, linear solvers, steps, and output machinery. Check if a capability exists in EdelweissFE before implementing it here.
- **Material Ownership (Thread Safety)**: A material point carries its state variables as mutable state. Never share a single material instance across multiple stencils. Use `MaterialProvider` (`edelweissfd/materials/provider.py`) to instantiate an isolated material per stencil.
- **Compiled Kernel Fallback**: When delegating to EdelweissFE's compiled Cython kernels, always provide an exact pure-Python fallback path. If a configuration or stress state is unsupported by the compiled kernel, return `None` from `_createCompiledKernel()` to gracefully fall back to Python.
- **Voigt Notation**: All stress and strain vectors follow **Marmot's** Voigt order `(11, 22, 33, 12, 13, 23)` with engineering shear strain $\gamma_{ij} = 2\varepsilon_{ij}$ (deliberately distinct from `edelweissfe.utils.voigtnotation`).
- **DOF Ordering & Flux Sign**: DOFs are ordered **node-major, field-minor**. Internal flux $P$ is positive sense ($R = -P + P_{\text{ext}}$), and tangent is $K = \partial P/\partial U$.

---

## License

By contributing to EdelweissFD, you agree that your contributions will be licensed under the **GNU Lesser General Public License v2.1** (LGPL-2.1). See [LICENSE.md](LICENSE.md) for full details.
