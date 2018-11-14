========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|

..    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/pydiffmap/badge/?version=master
    :target: http://pydiffmap.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/DiffusionMapsAcademics/pyDiffMap.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/DiffusionMapsAcademics/pyDiffMap

.. |codecov| image:: https://codecov.io/gh/DiffusionMapsAcademics/pyDiffMap/branch/master/graph/badge.svg
    :alt: Coverage Status
    :target: https://codecov.io/gh/DiffusionMapsAcademics/pyDiffMap

.. .. |commits-since| image:: https://img.shields.io/github/commits-since/DiffusionMapsAcademics/pyDiffMap/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/DiffusionMapsAcademics/pyDiffMap/compare/v0.1.0...master

.. .. |version| image:: https://img.shields.io/pypi/v/pyDiffMap.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/pyDiffMap

.. .. |commits-since| image:: https://img.shields.io/github/commits-since/DiffusionMapsAcademics/pyDiffMap/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/DiffusionMapsAcademics/pyDiffMap/compare/v0.1.0...master

.. .. |wheel| image:: https://img.shields.io/pypi/wheel/pyDiffMap.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/pyDiffMap

.. .. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pyDiffMap.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/pyDiffMap

.. .. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pyDiffMap.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/pyDiffMap


.. end-badges

This is the home of the documentation for pyDiffMap, an open-source project to develop a robust and accessible diffusion map code for public use. Our documentation is currently under construction, please bear with us. 

* Free software: MIT License.

Installation
============

Pydiffmap is installable using pip.  You can install it using the command

::

    pip install pyDiffMap

You can also install the package directly from the source directly by downloading the package from github and running the command below, optionally with the "-e" flag for an editable install.

::
    
    pip install [source_directory]

Documentation
=============

https://pyDiffMap.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

If you don't have tox installed, you can also run the python tests directly with 

::
    
    pytest

