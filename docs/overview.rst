Overview
========

Bayez is a python package for Bayesian redshift estimation, developed for the
`DESI project <http://desi.lbl.gov>`__.

The overall goals of this software project are to:

* Develop a generic method of redshift estimation that can be applied to any class of spectroscopic object for any redshift survey.
* Formulate a Bayesian estimator to full exploit astrophysical priors and provide a true redshift posterior probability distribution that can be used for subsequent science analysis.
* Implement an estimator whose inner-loop consists primarily of dense array operations, so that the code can take advantage of the single-threaded optimizations provided by `numba <http://numba.pydata.org>`__ and similar tools, and so that the algorithm is suitable for deployment on GPU architectures.

For a general introduction to the modules and function provided by this package, see the `examples notebook <https://github.com/dkirkby/bayez/blob/master/docs/nb/BayezExamples.ipynb>`__.  For results based on the DESI Redshift Data Challenge, see the `results notebook <https://github.com/dkirkby/bayez/blob/master/docs/nb/BayezResults.ipynb>`__.  For details of the mathematical formalism, see the latex note under ``docs/tex/``.
