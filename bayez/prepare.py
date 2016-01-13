# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Entry point to prepare priors.

The following environment variables are required (values are for my laptop):

export DESI_BASIS_TEMPLATES=/Data/DESI/basis_templates/v1.1
export DESISIM=/Users/david/Cosmo/DESI/code/desisim
export DESIMODEL=/Users/david/Cosmo/DESI/code/desimodel
export BAYEZ_DATA=/Data/DESI/bayez
"""
from __future__ import print_function, division

import os
import os.path

from astropy.utils.compat import argparse

import bayez

Samplers = {
    'qso': bayez.sampler.QSOSampler,
    'lrg': bayez.sampler.LRGSampler,
    'elg': bayez.sampler.ELGSampler,
}

def prepare(args=None):
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action = 'store_true',
        help = 'provide verbose output on progress')
    parser.add_argument('--classname', choices=['qso', 'lrg', 'elg'],
        default=None, help='Spectral class to prepare.')
    parser.add_argument('-n', '--num-kilo-spectra', default=1, metavar='K',
        help='Number of thousands of spectra to sample for the prior.')
    parser.add_argument('--downsampling', type=int, default=4, metavar='DS',
        help='Downsampling of 1A simulated pixels to use.')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
        help='Random seed to use for sampling templates.')
    parser.add_argument('--print-interval', type=int, default=500, metavar='P',
        help='Print messages for every P sampled spectra.')
    args = parser.parse_args(args)

    if args.classname is None:
        print('You must specify a spectral class.')
        return -1
    sampler = Samplers[args.classname]()
    if args.verbose:
        sampler.print_summary()

    simulator = bayez.simulation.Simulator(
        analysis_downsampling=args.downsampling, verbose=args.verbose)

    prior = bayez.prior.build_prior(
        args.classname, sampler, simulator, 1000 * args.num_kilo_spectra,
        seed=args.seed, print_interval=args.print_interval)

    path = os.environ.get('BAYEZ_DATA', '.')
    name = os.path.join(path, '{}_{}_{}k.fits'.format(
        args.classname, args.downsampling, args.num_kilo_spectra))
    if args.verbose:
        print('Saving prior to {}.'.format(name))
    prior.save(name, clobber=True)
