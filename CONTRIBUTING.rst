Thank you for considering improving `Empulse`, any contribution is much welcome!

.. _minimal reproducible example: https://stackoverflow.com/help/mcve
.. _open a new issue: https://github.com/ShimantoRahman/empulse/issues/new
.. _open a pull request: https://github.com/ShimantoRahman/empulse/compare
.. _empulse: https://github.com/ShimantoRahman/empulse
.. _uv: https://docs.astral.sh/uv/getting-started/installation/
.. _Conventional Commits: https://www.conventionalcommits.org/en/v1.0.0/


Asking questions
----------------

If you have any question about `Empulse`, if you are seeking for help,
or if you would like to suggest a new feature, you are encouraged to `open a new issue`_ so we can discuss it.
Bringing new ideas and pointing out elements needing clarification allows to make this library always better!


Reporting a bug
---------------

If you encountered an unexpected behavior using `Empulse`,
please `open a new issue`_ and describe the problem you have spotted.
Be as specific as possible in the description of the trouble so we can easily analyse it and quickly fix it.

An ideal bug report includes:

* The Python version you are using
* The `empulse` version you are using (you can find it with ``print(empulse.__version__)``)
* The `scikit-learn` version you are using (``print(sklearn.__version__)``)
* Your operating system name and version (Linux, MacOS, Windows)
* Your development environment and local setup (IDE, Terminal, project context, any relevant information that could be useful)
* Some `minimal reproducible example`_

Implementing changes
--------------------

If you are willing to enhance `Empulse` by implementing non-trivial changes,
please `open a new issue`_ first to keep a reference about why such modifications are made
(and potentially avoid unneeded work).

You will need:

* Python 3.11 or newer.
* `uv`_, which manages the development environment and the Python interpreters.
* A C compiler, because `Empulse` contains Cython extensions that are compiled when it is installed:
  the Microsoft C++ Build Tools on Windows, the Xcode command line tools on macOS, or ``gcc`` on Linux.

Then, the workflow would look as follows:

1. Fork the `empulse`_ project from GitHub.
2. Clone the repository locally::

    $ git clone git@github.com:your_name_here/empulse.git
    $ cd empulse

3. Install `Empulse` with its development dependencies::

    $ uv sync

   This creates a ``.venv`` virtual environment in the repository, installs `Empulse` into it in editable mode,
   and compiles the Cython extensions. Run commands inside it with ``uv run``, or activate ``.venv`` yourself.
   Changes to Python files take effect immediately; after editing a ``.pyx`` or ``.pxd`` file, recompile with::

    $ uv sync --reinstall-package empulse

4. Install the pre-commit hooks that will check your commits::

    $ uv run pre-commit install --install-hooks

5. Create a new branch from ``main``::

    $ git switch main
    $ git switch -c fix_bug

6. Implement the modifications wished. The code is formatted and linted with ruff (line length 120, single quotes)
   and docstrings follow the numpydoc style. The pre-commit hooks apply the formatting when you commit.
   Public code is type checked with mypy::

    $ uv run mypy empulse

7. Add tests under ``tests/`` (don't hesitate to be exhaustive!). Mark a test that takes long with
   ``@pytest.mark.slow``, and one that downloads data with ``@pytest.mark.remote``.
   While you work, run everything except those in parallel::

    $ uv run pytest -m "not slow and not remote" -n auto

   Before opening a pull request, run the complete suite, including the slow tests and the doctests,
   in a clean environment::

    $ uvx --with tox-uv tox -e py312-tests -- -m "not remote"

   The pull request is then tested on every supported Python version.

8. Remember to update the documentation if required. Code examples in the documentation and in docstrings
   are executed as tests, and the documentation must build without warnings::

    $ uv run sphinx-build -b html -W --keep-going docs docs/_build/html

9. If your development modifies `Empulse` behavior, add an entry to the ``Unreleased`` section of ``CHANGELOG.rst``.
   Each entry starts with one of the markers ``|MajorFeature|``, ``|Feature|``, ``|Enhancement|``,
   ``|Efficiency|``, ``|Fix|`` or ``|API|``.
10. ``add`` and ``commit`` your changes, then ``push`` your local project.
    Commit messages follow `Conventional Commits`_::

     $ git add .
     $ git commit -m "fix(metrics): short description of what changed"
     $ git push origin fix_bug

11. If the previous step failed due to the pre-commit hooks, fix the reported errors and try again.
12. Finally, `open a pull request`_ before getting it merged!
