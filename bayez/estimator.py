# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Support for redshift estimation.
"""
from __future__ import print_function, division

import time

import numpy as np

import astropy.table

import numba
from numba import float64, float32

@numba.vectorize([float64(float32, float32, float64, float32)])
def calculate_pull(ivar, flux, norm, template):
    return ivar * (flux - norm * template) ** 2

class RedshiftEstimator(object):

    def __init__(self, prior, dz=0.001):

        self.prior = prior

        # Initialize the posterior binning in redshift.
        num_z_bins = int(np.ceil((prior.z_max - prior.z_min) / dz))
        self.z_bin_edges = np.linspace(
            prior.z_min, prior.z_min + num_z_bins * dz, num_z_bins + 1)
        self.z_grid = 0.5 * (self.z_bin_edges[:-1] + self.z_bin_edges[1:])

        # Look up which redshift bin each spectrum of the prior occupies.
        self.zbin = np.digitize(prior.z, self.z_bin_edges)
        assert np.all((self.zbin > 0) & (self.zbin < len(self.z_bin_edges)))
        self.zbin -= 1

        # Pre-allocate large arrays.
        num_priors, num_pixels = prior.flux.shape
        num_mag_bins = len(prior.mag_grid)
        self.posterior = np.empty((num_z_bins,), dtype=np.float64)
        self.chisq = np.empty((num_priors, num_mag_bins), dtype=np.float64)
        self.pulls = np.empty((num_mag_bins, num_pixels), dtype=np.float64)

    def run(self, flux, ivar):

        # Loop over spectrum in the prior.
        for i, prior_flux in enumerate(self.prior.flux):
            # Calculate the normalization at each point of our magnitude grid
            # for this spectrum.
            flux_norm = 10 ** (-0.4 * (self.prior.mag_grid - self.prior.mag[i]))
            # Calculate the chisq for this template at each flux normalization
            '''
            # Version 1: simple expression (with lots of temporaries)
            pulls[:] = ivar * (flux - flux_norm[:, np.newaxis] * prior_flux)**2
            '''
            '''
            # Version 2: use ufuncs to eliminate temporaries
            pulls[:] = prior_flux
            pulls *= flux_norm[:, np.newaxis]
            pulls -= flux
            pulls = np.square(pulls, out=pulls)
            pulls *= ivar
            '''
            # Version 3: use numba ufunc with broadcasting
            calculate_pull(ivar, flux, flux_norm[:, np.newaxis], prior_flux, out=self.pulls)

            self.chisq[i] = np.sum(self.pulls, axis=-1)

        # Subtract the minimum chisq so that exp(-chisq/2) does not underflow
        # for the most probable bins.
        self.chisq_min = np.min(self.chisq)
        self.chisq -= self.chisq_min

        # Marginalize over magnitude for each prior spectrum.
        self.marginalized = np.sum(np.exp(-0.5 * self.chisq), axis=-1)
        self.marginalized /= np.sum(self.marginalized)

        # Find which template has the highest probability.
        self.i_best = np.argmax(self.marginalized)
        self.z_best = self.prior.z[self.i_best]

        # Calculate the mean redshift over the posterior.
        self.z_mean = np.average(self.prior.z, weights=self.marginalized)

        # Calculate the posterior probability in bins of redshift.
        self.posterior[:] = np.bincount(self.zbin, weights=self.marginalized,
                                        minlength=self.posterior.size)


def estimate_one(estimator, sampler, simulator, seed=1, i=0):
    """Run the estimator for a single simulated sample.

    This method requires that matplotlib is installed.
    """
    import matplotlib.pyplot as plt
    # Pick a random template to simulate.
    generator = np.random.RandomState((seed, i))
    true_flux, mag_pdf, true_z, true_mag, t_index = sampler.sample(generator)
    print('Generated [{}] z = {:.4f}, mag = {:.2f}'
        .format(t_index, true_z, true_mag))

    # Plot the template before simulation and without noise.
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.fill_between(sampler.obs_wave, true_flux, 0., color='green', alpha=0.2)
    plt.plot(sampler.obs_wave, true_flux, 'g-',
        label='z={:.2f},mag={:.2f}'.format(true_z, true_mag))
    plt.xlim(sampler.obs_wave[0], sampler.obs_wave[-1])
    plt.ylim(0., 2 * np.max(true_flux))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.legend()

    # Simulate the template.
    results = simulator.simulate(
        sampler.obs_wave, true_flux, noise_generator=generator)

    # Run the estimator on the simulated analysis pixels.
    start_time = time.time()
    estimator.run(simulator.flux, simulator.ivar)
    elapsed = time.time() - start_time
    print('Elapsed time {:.3f}s'.format(elapsed))
    print('MAP: z[{}] = {:.4f}'.format(estimator.i_best, estimator.z_best))
    print('<z> = {:.4f}'.format(estimator.z_mean))

    # Plot the posterior probability distribution centered on the true value.
    plt.subplot(1, 2, 2)
    plt.hist(estimator.z_grid, weights=estimator.posterior,
        bins=estimator.z_bin_edges, histtype='stepfilled', alpha=0.25)
    plt.axvline(true_z, ls='-', color='red')
    plt.axvline(estimator.z_mean, ls='--', color='red')
    plt.xlabel('Redshift $z$')
    plt.ylabel('Posterior $P(z|D)$')
    plt.xlim(true_z - 0.02, true_z + 0.02)

    plt.tight_layout()
    plt.show()

def estimate_batch(estimator, num_batch, sampler, simulator,
                   seed=1, print_interval=500):

    results = astropy.table.Table(
        names = ('i', 't_index', 'mag', 'z', 'dz_map', 'dz_avg'),
        dtype = ('i4', 'i4', 'f4', 'f4', 'f4', 'f4')
    )
    for i in xrange(num_batch):
        generator = np.random.RandomState((seed, i))
        true_flux, mag_pdf, true_z, true_mag, t_index = sampler.sample(generator)
        simulator.simulate(sampler.obs_wave, true_flux, noise_generator=generator)
        estimator.run(simulator.flux, simulator.ivar)
        results.add_row(dict(
            i=i, t_index=t_index, mag=true_mag, z=true_z,
            dz_map=estimator.z_best - true_z,
            dz_avg=estimator.z_mean - true_z
        ))

        if print_interval and (i + 1) % print_interval == 0:
            print('[{}] mag = {:.2f}, z = {:.2f}, dz = {:+.04f}, {:+.04f}'
                .format(*results[i]))

    return results
