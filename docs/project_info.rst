===================
Project Information
===================

How Empulse changes between releases, how to take part in it, and the terms it is distributed
under.

.. grid:: 1 3 3 3
    :gutter: 3

    .. grid-item-card:: :octicon:`versions;1.5em;sd-mr-1` Changelog
        :link: project/changelog
        :link-type: doc

        Every release, with the new features, fixes and API changes it carried.

    .. grid-item-card:: :octicon:`git-pull-request;1.5em;sd-mr-1` Contributing
        :link: project/contributing
        :link-type: doc

        Asking a question, reporting a bug, and the workflow for implementing a change yourself.

    .. grid-item-card:: :octicon:`law;1.5em;sd-mr-1` License
        :link: project/license
        :link-type: doc

        Empulse is distributed under the MIT License. The full text.

.. Listed explicitly rather than with :glob:, so that the cards above cannot silently fall out of
   step with the pages: a new page under project/ that nobody adds here is an orphan, which the
   ``-W`` documentation build reports rather than quietly leaving out of the grid.

.. toctree::
    :maxdepth: 2
    :hidden:

    project/changelog.rst
    project/contributing.rst
    project/license.rst