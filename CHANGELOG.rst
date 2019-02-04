
Changelog
=========

0.2.0.1 (2019-02-04)
New Features
~~~~~~~~~~~~
* Added a more generous epsilon procedure for convenience.

0.2.0 (2019-02-01)
------------------

New Features
~~~~~~~~~~~~
* Added support for user-provided kernel functions. 
* Added a utility for building a sparse matrix from a function on the data.
* (Re)added separate TMDmap class wrapping base diffusion map class to 
  allow for easier construction of TMDmaps. 
* Added ability to explicitly provide the sampled density for q^alpha normalization.
* Added Variable Bandwidth Diffusion Maps.
* Added a new out-of-sample extension method that should work for variable bandwidth methods.

Tweaks and Modifications
~~~~~~~~~~~~~~~~~~~~~~~~
* Changed from exp^(-d^2) to exp^(-d^2/4) convention.
* Moved weight functionality into a function provided on initialization, 
  rather than input values, and added a helper function that allows values to
  be read from a lookup table.
* Improved the Diffusion Map test suite.
* Moved out-of-sample routines into separate functions.
* Moved matrix symmetrization into newly made utility file.
* Adjusted constructor for the diffusion map to take the kernel object directly.

Bugfixes
~~~~~~~~
* Fixed bug where weight matrices were not included for out of sample extension.

Other
~~~~~
* Moved to MIT license.

0.1.0 (2017-12-06)
------------------

* Fixed setup.py issues.

0.1.0 (2017-12-06)
------------------

* Added base functionality to the code.
