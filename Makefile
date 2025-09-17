.PHONY: clean build help test

clean:
	powershell Remove-Item -Recurse -Force dist\*
	powershell Remove-Item -Recurse -Force build\*
	powershell Remove-Item -Recurse -Force *.egg-info\

build: clean
	uv build

upload: build
	uvx twine upload dist/*

compile:
	uv sync --reinstall-package empulse

test:
	uv run pytest --cov-report term --cov=empulse tests/
	uv run coverage html
	start chrome "%CD%\htmlcov\index.html"

doctest:
	uv run pytest --doctest-modules empulse/

tox:
	uvx --with tox-uv tox -e py312-lint
	uvx --with tox-uv tox -e py312-docs
	uvx --with tox-uv tox -f tests

lint:
	uvx ruff format --preview
	uvx ruff check --fix --preview
	uvx ruff format --preview

typecheck:
	uvx mypy empulse

pre-commit:
	uvx pre-commit run --all-files

# Content from the docs directory Makefile
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# "make html" will build the html docs
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


help:
	@echo - clean:	remove all build artifacts
	@echo - build:	build the project
	@echo - upload:	upload the project to pypi
	@echo - html:		build the html docs
	@echo - test:		run the tests
	@echo - doctest:	run the doctests
	@echo - lint:		run the linter
	@echo - pre-commit:	run the pre-commit checks
