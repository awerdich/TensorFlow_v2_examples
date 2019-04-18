========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/TensorFlow_v2_examples/badge/?style=flat
    :target: https://readthedocs.org/projects/TensorFlow_v2_examples
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/awerdich/TensorFlow_v2_examples.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/awerdich/TensorFlow_v2_examples

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/awerdich/TensorFlow_v2_examples?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/awerdich/TensorFlow_v2_examples

.. |version| image:: https://img.shields.io/pypi/v/tensorflow-v2-examples.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/tensorflow-v2-examples

.. |commits-since| image:: https://img.shields.io/github/commits-since/awerdich/TensorFlow_v2_examples/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/awerdich/TensorFlow_v2_examples/compare/v0.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/tensorflow-v2-examples.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/tensorflow-v2-examples

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/tensorflow-v2-examples.svg
    :alt: Supported versions
    :target: https://pypi.org/project/tensorflow-v2-examples

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/tensorflow-v2-examples.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/tensorflow-v2-examples


.. end-badges

An example package. Generated with cookiecutter-pylibrary.

* Free software: MIT license

Installation
============

::

    pip install tensorflow-v2-examples

Documentation
=============


https://TensorFlow_v2_examples.readthedocs.io/


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
