
Changelog
=========

0.2.0 (2018-??-??)
------------------

New Features
~~~~~~~~~~~~
* Added support for user-provided kernel functions. 
* Added support for general weight matrices that take both points as input.
* Added a utility for building a sparse matrix from a function on the data.
* (Re)added separate TMDmap class wrapping base diffusion map class to 
  allow for easier construction of TMDmaps. 

Tweaks and Modifications
~~~~~~~~~~~~~~~~~~~~~~~~
* Changed from exp^(-d^2) to exp^(-d^2/4) convention.
* Moved weight functionality into a function provided on initialization, 
  rather than input values, and added a helper function that allows values to
  be read from a lookup table.

Bugfixes
~~~~~~~~
* Fixed bug where weight matrices were not included for out of sample extension.

0.1.0 (2017-12-06)
------------------

* Fixed setup.py issues.

0.1.0 (2017-12-06)
------------------

* Added base functionality to the code.
