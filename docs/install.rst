Installation
============

Install the latest stable release using::

    pip install bayez

Alternatively, you can install the latest developer version from github::

    github clone https://github.com/dkirkby/bayez.git
    cd bayez
    python setup.py install

Dependencies
------------

Bayez has the following package requirements:

* `numpy <http://www.numpy.org/>`__
* `scipy <http://www.scipy.org/>`__
* `astropy <http://www.astropy.org/>`__
* `numba <http://numba.pydata.org>`__

In addition, bayez requires external models for the astrophysical priors. Currently these are provided by the `desisim <https://github.com/desihub/desisim>`__ package, which in turns requires the following `DESI Project <http://desi.lbl.gov>`__ packages:

* `specter <https://github.com/desihub/specter>`__
* `desispec <https://github.com/desihub/desispec>`__
* `desimodel <https://github.com/desihub/desimodel>`__

Note that the current version of `desimodel` is still under SVN, but will migrate to github soon. The bayez package also requires an external spectrograph simulation, for which it uses the `specsim <https://github.com/desihub/specsim>`__ package.
