# use cmd.exe instead of sh:
set shell := ["cmd.exe", "/c"]

# Default recipe to display help
[private]
default:
    @just --list

# Remove all build artifacts
[windows]
[group('deploy')]
clean:
    powershell Remove-Item -Recurse -Force dist\*
    powershell Remove-Item -Recurse -Force build\*
    powershell Remove-Item -Recurse -Force *.egg-info\

# Remove all build artifacts
[unix]
[group('deploy')]
clean:
    rm -rf dist/*
    rm -rf build/*
    rm -rf *.egg-info

# Build the project
[group('deploy')]
build: clean
    uv build

# Upload the project to PyPI
[group('deploy')]
upload: build
    uvx twine upload dist/*

# Compile/reinstall the package
[group('deploy')]
compile:
    uv sync --reinstall-package empulse

# Run pytest tests (optionally specify: models, metrics, or run all by default)
[group('test')]
test target='':
    uv run pytest tests/{{target}}

_cov:
    uv run pytest --cov-report term --cov=empulse tests/
    uv run coverage html

# Run tests with coverage
[windows]
[group('test')]
cov: _cov
    start chrome %CD%\htmlcov\index.html

# Run tests with coverage
[macos]
[group('test')]
cov: _cov
    open htmlcov/index.html

# Run tests with coverage
[linux]
[group('test')]
cov: _cov
    xdg-open htmlcov/index.html || echo "Coverage report generated at htmlcov/index.html"

# Run doctests
[group('test')]
doctest:
    uv run pytest --doctest-modules empulse/

# Run tox tests
[group('test')]
tox:
    uvx --with tox-uv tox -e py312-lint
    uvx --with tox-uv tox -e py312-docs
    uvx --with tox-uv tox -f tests

# Run linter and formatter
[group('lint')]
lint:
    uvx ruff format --preview
    uvx ruff check --fix --preview
    uvx ruff format --preview

# Run type checker
[group('lint')]
type:
    mypy empulse

# Run pre-commit checks
[group('lint')]
pre-commit:
    uvx pre-commit run --all-files

# Sphinx documentation variables
SPHINXOPTS := ""
SPHINXBUILD := "sphinx-build"
SOURCEDIR := "docs"
BUILDDIR := "docs/_build"

# Build HTML documentation
[group('docs')]
html:
    {{SPHINXBUILD}} -M html {{SOURCEDIR}} {{BUILDDIR}} {{SPHINXOPTS}}
    start chrome %CD%\{{BUILDDIR}}\index.html

# Build documentation in other formats (e.g., just latex, just epub, etc.)
[positional-arguments]
[group('docs')]
sphinx-build target:
    {{SPHINXBUILD}} -M {{target}} {{SOURCEDIR}} {{BUILDDIR}} {{SPHINXOPTS}}

# Check all if links in docs are valid
[group('docs')]
linkcheck:
    {{SPHINXBUILD}} -M linkcheck {{SOURCEDIR}} {{BUILDDIR}} {{SPHINXOPTS}}

# Verify version consistency across __init__.py, CITATION.cff, and CHANGELOG.rst
[windows]
[group('deploy')]
verify-version:
    @powershell -ExecutionPolicy Bypass -File scripts/verify-version.ps1

# Run all preflight checks before deployment
[windows]
[group('deploy')]
preflight:
    just verify-version
    just linkcheck
    just tox
