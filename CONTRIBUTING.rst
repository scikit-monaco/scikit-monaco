
Contributing
============

Scikit-monaco is in its infancy. There is a lot of scope for developpers to make
a strong impact by contributing code, documentation and expertise. All
contributions are welcome and, if you are unsure how to contribute or are
unfamiliar with scipy and numpy, we will happily mentor you. Just send an email
to `<pascal@bugnion.org>`.

How to contribute
-----------------

The `documentation <http://docs.scipy.org/doc/numpy/dev/gitwash/index.html>`_ for Numpy gives a detailed description of how to contribute. Most of this information applies to development for ``scikit-monaco``.

Developping with git
^^^^^^^^^^^^^^^^^^^^

You will need the `Git version control system <http://git-scm.com>`_ and an account on `Github <https://github.com>`_ to
contribute to scikit-monaco.

1. Fork the `project repository <http://github.com/scikit-monaco/scikit-monaco>`_ by clicking `Fork` in the top right of the page. This will create a copy of the fork under your account on Github.

2. Clone the repository to your computer::
   
    $ git clone https://github.com/YourUserID/scikit-monaco.git

3. Install scikit-monaco by running::

    $ python setup.py install

   in the package's root directory. You should now be able to test the code
   by running ``$ nosetests`` in the package's root directory.


You can now make changes and contribute them back to the source code:

1. Create a branch to hold your changes::

    $ git checkout -b my-feature

   and start making changes.

2. Work on your local copy. When you are satisfied with the changes, commit
   them to your local repository::

    $ git add modified files
    $ git commit

   You will be asked to write a commit message. Explain the reasoning behind
   the changes that you made.

3. Propagate the changes back to your github account::

    $ git push -u origin my-feature

4. To integrate the changes into the main code repository, click `Pull Request`
   on the `scikit-quantum` repository page on your accont. This will notify the
   committers who will review your code.

Updating your repository
^^^^^^^^^^^^^^^^^^^^^^^^

To keep your private repository updated, you should add the main repository as 
a remote::
    
    $ git remote add upstream git://github.com/scikit-monaco/scikit-monaco.git

To update your private repository, you can then fetch new commits from
upstream::

    $ git fetch upstream
    $ git rebase upstream/master


Guidelines
----------

Workflow
^^^^^^^^

We loosely follow the `git workflow <http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html>`_ used in numpy development.  Features should
be developped in separate branches and merged into the master branch when
complete. Avoid putting new commits directly in your ``master`` branch.


Code
^^^^

Please follow the `PEP8 conventions <http://www.python.org/dev/peps/pep-0008/>`_ for formatting and indenting code and for variable names.


Documentation
^^^^^^^^^^^^^

Scikit-quantum uses `sphinx <http://sphinx-doc.org/>`_ with `numpydoc <https://pypi.python.org/pypi/numpydoc>`_ to process the documentation. We
follow the `numpy
convention <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ on writing docstrings.

Use ``make html`` in the ``doc`` folder to build the documentation.


Testing
^^^^^^^

We use `nose <http://nose.readthedocs.org/en/latest/>`_ to test
`scikit-quantum`. Running ``nosetests`` in the root directory of the package
will run the tests.

