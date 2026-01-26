# Pre-commit Hooks Setup

This document describes the pre-commit hooks configured for the VibeAgent project.

## Overview

Pre-commit hooks are automated checks that run before each commit to ensure code quality and consistency. They help catch issues early in the development process.

## Installation

Pre-commit hooks are automatically installed when you run:

```bash
pre-commit install
```

The hooks will then run automatically on every `git commit` command.

## Running Hooks Manually

To run all hooks on all files:
```bash
pre-commit run --all-files
```

To run a specific hook:
```bash
pre-commit run <hook-id> --all-files
```

## Configured Hooks

### 1. Basic File Checks (`pre-commit/pre-commit-hooks`)
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml**: Validates YAML syntax
- **check-toml**: Validates TOML syntax
- **check-json**: Validates JSON syntax
- **debug-statements**: Checks for debug statements (breakpoint, pdb, etc.)
- **check-merge-conflict**: Checks for merge conflict markers
- **check-added-large-files**: Prevents adding large files (>1MB)
- **check-ast**: Checks Python syntax
- **check-builtin-literals**: Checks for proper use of literals
- **check-docstring-first**: Checks docstrings come first
- **check-executables-have-shebangs**: Checks executables have shebangs
- **check-shebang-scripts-are-executable**: Checks shebang scripts are executable
- **check-symlinks**: Checks for broken symlinks
- **destroyed-symlinks**: Checks for symlink changes
- **fix-byte-order-marker**: Removes byte order marker
- **mixed-line-ending**: Normalizes line endings to LF
- **name-tests-test**: Ensures test files follow naming convention

### 2. Ruff (Linting and Formatting)
- **ruff**: Fast Python linter with auto-fix
- **ruff-format**: Code formatter

Configuration in `pyproject.toml`:
- Target Python 3.10+
- Line length: 100 characters
- Enforces imports sorting, type annotations, security checks
- Excludes `__pycache__`, virtual environments, and archives
- Ignores ISC001 (conflicts with formatter)
- Allows assert statements in test files

### 3. MyPy (Type Checking)
- **mypy**: Static type checker

Configuration:
- Strict mode enabled
- Checks type annotations
- Excludes tests and scripts from strict typing
- Includes pydantic and sqlalchemy plugins

### 4. PyProject Validation
- **validate-pyproject**: Validates pyproject.toml syntax

### 5. GitHub Configuration Checks
- **check-github-workflows**: Validates GitHub Actions workflows
- **check-dependabot**: Validates Dependabot configuration

## Exclusions

The following paths are excluded from hooks:
- `__pycache__/`
- `.venv/`, `venv/`
- `.git/`
- `build/`, `dist/`
- `archives/`
- CSV and TSV files (for trailing whitespace)

## Test Files

Test files have special permissions:
- Allowed to use assert statements (`S101`)
- Type annotations are not strictly enforced
- Can use print statements for debugging

## Troubleshooting

### Hook installation issues
```bash
pre-commit clean
pre-commit install --force
```

### Bypassing hooks (not recommended)
```bash
git commit --no-verify -m "Your message"
```

### Skipping specific hooks
```bash
SKIP=mypy git commit -m "Your message"
```

## CI Integration

Pre-commit hooks are designed to work with pre-commit.ci for automated fixes on PRs. Configuration includes:
- Auto-fixing of linting issues
- Weekly dependency updates
- Consistent commit messages

## Performance

Hooks are optimized for performance:
- Ruff runs in parallel for speed
- MyPy uses incremental checking
- Large files and virtual environments are excluded

## Development Workflow

1. Make your changes
2. Stage files with `git add`
3. Run `pre-commit run --all-files` to check everything
4. Fix any issues reported
5. Commit with `git commit`
6. Hooks will run automatically and prevent commit if issues are found

## Further Reading

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
