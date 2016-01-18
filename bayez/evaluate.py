# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Entry point to evaluate the redshift estimator on simulated spectra.

The following environment variables are required (values are for my laptop):

export BAYEZ_DATA=/Data/DESI/bayez

Example usage:

eval_bayez --prior qso_4_1k.fits -n 100 --seed 10 --verbose
"""
from __future__ import print_function, division

import os
import os.path

from astropy.utils.compat import argparse

import bayez

def evaluate(args=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action = 'store_true',
        help = 'provide verbose output on progress')
    parser.add_argument('--prior', default=None,
        help='Name of the prior to use, e.g., qso-4-10k.fits')
    parser.add_argument('-n', '--num-spectra', type=int, default=1, metavar='N',
        help='Number of spectra to use for evaluation.')
    parser.add_argument('--mag-err', type=float, default=0.1, metavar='dM',
        help='RMS error on targeting magnitudes to simulate.')
    parser.add_argument('--quad-order', type=int, default=16, metavar='N',
        help='Quadrature order to use for magnitude marginalization.')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
        help='Random seed to use for sampling templates.')
    args = parser.parse_args(args)

    if args.prior is None:
        print('You must specify a prior to use.')
        return -1
    if args.seed is None:
        print('You must specify a seed to use.')
        return -1

    # Parse the prior name, which is expected to have format xxx-nn-nnnk.fits
    basename, ext = os.path.splitext(args.prior)
    if ext != '.fits':
        print('Unexpected extension for prior filename: {}'.format(ext))
        return -1
    try:
        classname, downsampling, _ = basename.split('_')
        downsampling = int(downsampling)
    except ValueError:
        print('Badly formatted prior filename: {}.'.format(basename))
        return -1
    if classname not in ('qso', 'lrg', 'elg', 'star'):
        print('Invalid prior class name: {}.'.format(classname))
        return -1
    if args.verbose:
        print('Prior uses downsampling {} for class {}.'
            .format(downsampling, classname))

    # Load the prior now.  Prepend $BAYEZ_DATA unless we already have
    # an absolute path.
    path = os.environ.get('BAYEZ_DATA', '.')
    if not os.path.isabs(args.prior):
        args.prior = os.path.join(path, args.prior)
    if args.verbose:
        print('Reading prior from {}'.format(args.prior))
    prior = bayez.prior.load_prior(args.prior)

    # Prepare to simulate spectra for evaluation.
    sampler = bayez.sampler.Samplers[classname]()
    simulator = bayez.simulation.Simulator(
        analysis_downsampling=downsampling, verbose=args.verbose)

    # Run the evaluation.
    estimator = bayez.estimator.RedshiftEstimator(
        prior, dz=0.001, quadrature_order=args.quad_order)
    results = bayez.estimator.estimate_batch(
        estimator, args.num_spectra, sampler, simulator, mag_err=args.mag_err,
        seed=args.seed, print_interval=500 if args.verbose else 0)

    # Save the results.
    name = os.path.join(path,
        '{}_q{:+d}_{}.fits'.format(basename, args.quad_order, args.seed))
    if args.verbose:
        print('Saving results to {}'.format(name))
    results.write(name, overwrite=True)
