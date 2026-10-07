# Windows runs recipes through cmd.exe, Linux and macOS through sh
set windows-shell := ["cmd.exe", "/c"]

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

# Run the tests in parallel, skipping the slow and remote ones (optionally specify: models, metrics, ...)
[group('test')]
fast-test target='':
    uv run pytest -m "not slow and not remote" -n auto tests/{{target}}

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

# Time the known hot paths (optionally name cases to run a subset)
[group('test')]
bench *cases:
    uv run python scripts/benchmark.py {{cases}}

# Run doctests
[group('test')]
doctest:
    uv run pytest --doctest-modules empulse/ --ignore=empulse/metrics/_loss --ignore=empulse/metrics/_cy_convex_hull --ignore=empulse/models/tree/_cstree --ignore=empulse/models/tree/_cy_proftree

# Run tox tests
[group('test')]
tox:
    uvx --with tox-uv tox -e py312-lint
    uvx --with tox-uv tox -e py312-docs
    uvx --with tox-uv tox -f tests

# Update scikit-learn compat tox environments by querying PyPI for latest patch releases
[group('test')]
update-sklearn-compat:
    uv run scripts/update_sklearn_compat.py

# Run all scikit-learn compatibility tox environments (auto-generated, run update-sklearn-compat first)
[group('test')]
sklearn-compat:
    uv run scripts/sklearn_compat_test_runner.py

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

# Regenerate the documentation figures (light and dark variants)
[group('docs')]
figures:
    uv run python scripts/figures/build.py

_html:
    {{SPHINXBUILD}} -M html {{SOURCEDIR}} {{BUILDDIR}} {{SPHINXOPTS}}

# Build HTML documentation
[windows]
[group('docs')]
html: _html
    start chrome %CD%\{{BUILDDIR}}\html\index.html

# Build HTML documentation
[macos]
[group('docs')]
html: _html
    open {{BUILDDIR}}/html/index.html

# Build HTML documentation
[linux]
[group('docs')]
html: _html
    xdg-open {{BUILDDIR}}/html/index.html || echo "Documentation built at {{BUILDDIR}}/html/index.html"

# Build HTML documentation, failing on any warning (matches Read the Docs)
[group('docs')]
html-strict:
    {{SPHINXBUILD}} -b html -W --keep-going {{SOURCEDIR}} {{BUILDDIR}}/html-strict {{SPHINXOPTS}}

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

# Verify version consistency across __init__.py, CITATION.cff, and CHANGELOG.rst
[unix]
[group('deploy')]
verify-version:
    @sh scripts/verify-version.sh

# Run all preflight checks before deployment
[group('deploy')]
preflight: verify-version html-strict linkcheck tox update-sklearn-compat sklearn-compat
