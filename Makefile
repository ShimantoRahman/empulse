.PHONY: clean build help test

clean:
	powershell Remove-Item -Recurse -Force dist\*
	powershell Remove-Item -Recurse -Force build\*
	powershell Remove-Item -Recurse -Force *.egg-info\

build: clean
	python setup.py sdist bdist_wheel

upload: build
	twine upload dist/*

test:
	pytest --cov-report term --cov=empulse tests/

doctest:
	pytest --doctest-modules empulse/

tox:
	tox -e py312-lint
	tox -e py312-docs
	tox -f tests -p

lint:
	ruff check --fix
	ruff format

typecheck:
	mypy empulse

pre-commit:
	pre-commit run --all-files

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
