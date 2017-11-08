========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |travis|

..    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. .. |docs| image:: https://readthedocs.org/projects/python-pydiffmap/badge/?style=flat
    :target: https://readthedocs.org/projects/python-pydiffmap
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/ZofiaTr/python-pydiffmap.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/ZofiaTr/python-pydiffmap

.. .. |codecov| image:: https://codecov.io/github/ZofiaTr/python-pydiffmap/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/ZofiaTr/python-pydiffmap

.. .. |version| image:: https://img.shields.io/pypi/v/pyDiffMap.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/pyDiffMap

.. .. |commits-since| image:: https://img.shields.io/github/commits-since/ZofiaTr/python-pydiffmap/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ZofiaTr/python-pydiffmap/compare/v0.1.0...master

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

Library for constructing variable bandwidth diffusion maps

* Free software: MIT license

Installation
============

::

    pip install pyDiffMap

Documentation
=============

https://python-pydiffmap.readthedocs.io/

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
