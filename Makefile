.PHONY: clean build

clean:
	powershell Remove-Item -Recurse -Force dist\*
	powershell Remove-Item -Recurse -Force build\*
	powershell Remove-Item -Recurse -Force *.egg-info\

build: clean
	python setup.py sdist bdist_wheel

upload: build
	twine upload dist/*

# Content from the docs directory Makefile
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# "make html" will build the html docs
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)