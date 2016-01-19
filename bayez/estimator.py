# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Support for redshift estimation.
"""
from __future__ import print_function, division

import time

import numpy as np

import scipy.special
import scipy.interpolate

import astropy.table

import numba
from numba import float64, float32

@numba.vectorize([float64(float32, float32, float64, float32)])
def calculate_pull(ivar, flux, norm, template):
    return ivar * (flux - norm * template) ** 2

class RedshiftEstimator(object):

    def __init__(self, prior, dz=0.001, quadrature_order=16, nsig=3.0):
        """
        A quadrature_error > 0 specifies Gauss-Hermite quadrature for
        marginalizing over magnitudes.  A value < 0 specifies trapezoidal
        quadrature over +/- nsig sigmas.  A value of 0 forces trapezoidal
        quadrature over the full range mag_min - mag_max, which is also
        used when mag_err = 0 is passed to run().
        """
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
        self.pulls = np.empty((num_mag_bins, num_pixels), dtype=np.float64)
        self.chisq = np.empty((num_priors, num_mag_bins), dtype=np.float64)

        if quadrature_order > 0:
            # Calculate abscissas and weights for Gauss-Hermite quadrature.
            hermite = scipy.special.hermite(quadrature_order)
            self.quadrature_xi, self.quadrature_wi = (
                hermite.weights[:, :2].transpose())
            self.quadrature_wi /= np.sqrt(np.pi)
        elif quadrature_order < 0:
            quadrature_order = -quadrature_order
            # Use equally spaced points spanning +/-3 sigma.
            self.quadrature_xi = (
                np.linspace(-nsig, +nsig, quadrature_order) / np.sqrt(2))
            self.quadrature_wi = (np.exp(-self.quadrature_xi**2) *
                2 * nsig / (quadrature_order - 1.) / np.sqrt(2 * np.pi))

        # We assume that the pulls and chisq arrays needed for
        # quadrature fit into the non-quadrature arrays.
        if quadrature_order > num_mag_bins:
            raise ValueError('Cannot have |quadrature_order| > num_mag_bins.')
        self.quadrature_order = quadrature_order

        # Create a linear interpolator for the magnitude prior P(m|i).
        self.mag_pdf_interpolator = scipy.interpolate.interp1d(
            self.prior.mag_grid, self.prior.mag_pdf, kind='linear',
            axis=-1, copy=False, bounds_error=False, fill_value=0.,
            assume_sorted=True)

    def run(self, flux, ivar, mag, mag_err):

        # Use Gauss-Hermite quadrature for the flux normalization integral
        # if we have an observed magnitude to localize the integrand.
        if mag_err > 0 and self.quadrature_order != 0:
            # Calculate the asbscissas to use.
            mj = np.sqrt(2) * mag_err * self.quadrature_xi + mag
            # Interpolate the magnitude prior P(m|i) onto these abscissas.
            weights = self.quadrature_wi * self.mag_pdf_interpolator(mj)
            # Use views
            pulls = self.pulls[:self.quadrature_order]
            chisq = self.chisq[:, :self.quadrature_order]
        else:
            # We will not include a photometric likelihood P(M|m,i) factor
            # so the weights are equal to the magnitude prior P(m|i) and
            # we marginalize over the full magnitude grid.
            mj = self.prior.mag_grid
            weights = self.prior.mag_pdf
            pulls = self.pulls
            chisq = self.chisq

        # Loop over spectrum in the prior.
        for i, prior_flux in enumerate(self.prior.flux):
            # Calculate the normalization at each point of our magnitude grid
            # for this spectrum.
            flux_norm = 10 ** (-0.4 * (mj - self.prior.mag[i]))
            # Calculate the chisq for this template at each flux normalization
            calculate_pull(ivar, flux, flux_norm[:, np.newaxis],
                           prior_flux, out=pulls)

            # Sum over pixels to calculate chisq = -2 log(L) for the
            # spectroscopic likelihood L=P(D|m_j,i).
            chisq[i] = np.sum(pulls, axis=-1)

        # Subtract the minimum chisq so that exp(-chisq/2) does not underflow
        # for the most probable bins.
        chisq_min = np.min(chisq)
        chisq -= chisq_min

        # Marginalize over magnitude for each prior spectrum to calculate
        # the posterior P(i|D,M).  The spectroscopic likelihood P(D|i) equals
        # exp(-chisq/2) and the weights are the product of the magnitude
        # prior P(m|i) and the magnitude likelihood P(M|m,dM), if any.
        self.marginalized = np.sum(np.exp(-0.5 * chisq) * weights, axis=-1)
        marginal_sum = np.sum(self.marginalized)
        if marginal_sum <= 0:
            raise RuntimeError('Posterior probability ~ 0.')
        self.marginalized /= marginal_sum

        # Find which template has the highest probability.
        self.i_best = np.argmax(self.marginalized)
        self.z_best = self.prior.z[self.i_best]

        # Calculate the mean redshift over the posterior.
        self.z_mean = np.average(self.prior.z, weights=self.marginalized)

        # Calculate the cummulative distribution function of the posterior.
        z_sort_order = np.argsort(self.prior.z)
        z_sorted = np.insert(self.prior.z[z_sort_order],
                             0, self.prior.z_min)
        p_sorted = np.insert(self.marginalized[z_sort_order], 0, 0.)
        cdf = np.cumsum(p_sorted)
        # Interpolate to estimate posterior quantiles.
        cdf_inv = scipy.interpolate.interp1d(cdf, z_sorted, kind='linear',
                copy=False, bounds_error=False, fill_value=1.,
                assume_sorted=True)
        self.z_limits = cdf_inv([0.025, 0.16, 0.50, 0.84, 0.975])

        # Calculate the posterior probability in bins of redshift.
        self.posterior[:] = np.bincount(self.zbin, weights=self.marginalized,
                                        minlength=self.posterior.size)


def estimate_one(estimator, sampler, simulator, seed=1, i=0, mag_err=0.1,
                 save=None):
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
    mag_obs = true_mag + mag_err * generator.randn()

    # Run the estimator on the simulated analysis pixels.
    start_time = time.time()
    estimator.run(simulator.flux, simulator.ivar, mag_obs, mag_err)
    elapsed = time.time() - start_time
    print('Elapsed time {:.3f}s'.format(elapsed))
    print('MAP: z[{}] = {:.4f}'.format(estimator.i_best, estimator.z_best))
    print('<z> = {:.4f}'.format(estimator.z_mean))

    # Plot the posterior probability distribution.
    plt.subplot(1, 2, 2)
    plt.hist(estimator.z_grid, weights=estimator.posterior,
        bins=estimator.z_bin_edges, histtype='stepfilled', alpha=0.25)
    plt.axvline(true_z, ls='-', color='red')
    plt.axvline(estimator.z_mean, ls='--', color='red')
    for z in estimator.z_limits:
        plt.axvline(z, ls=':', color='red')

    plt.xlabel('Redshift $z$')
    plt.ylabel('Posterior $P(z|D)$')
    z_pad = max(0.002, 0.1 * (estimator.z_limits[-1] - estimator.z_limits[0]))
    plt.xlim(estimator.z_limits[0] - z_pad, estimator.z_limits[-1] + z_pad)

    # Plot markers for each prior sample within the plot range.
    y_lo, y_hi = plt.gca().get_ylim()
    y_pos = 0.1 * y_lo + 0.9 * y_hi
    visible = ((estimator.prior.z > estimator.z_limits[0] - z_pad) &
               (estimator.prior.z < estimator.z_limits[-1] + z_pad))
    for z in estimator.prior.z[visible]:
        plt.plot(z, y_pos, 'rx', alpha=0.5)
    plt.ylim(y_lo, y_hi)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()

def estimate_batch(estimator, num_batch, sampler, simulator,
                   seed=1, mag_err=0.1, quadrature_order=16,
                   print_interval=500):

    results = astropy.table.Table(
        names = ('i', 't_true', 'mag', 'z', 'dz_map', 'dz_avg',
            'p_best', 't_best', 'z95_lo', 'z68_lo', 'z50', 'z68_hi', 'z95_hi'),
        dtype = ('i4', 'i4', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4',
            'f4', 'f4', 'f4', 'f4', 'f4')
    )
    num_failed = 0
    start_time = time.time()
    print('Starting at {}.'.format(time.ctime(start_time)))
    for i in xrange(num_batch):
        generator = np.random.RandomState((seed, i))
        true_flux, mag_pdf, true_z, true_mag, t_index = (
            sampler.sample(generator))
        simulator.simulate(
            sampler.obs_wave, true_flux, noise_generator=generator)
        mag_obs = true_mag + mag_err * generator.randn()
        try:
            estimator.run(simulator.flux, simulator.ivar, mag_obs, mag_err)
            results.add_row(dict(
                i=i, t_true=t_index, mag=true_mag, z=true_z,
                dz_map=estimator.z_best - true_z,
                dz_avg=estimator.z_mean - true_z,
                p_best=estimator.i_best,
                t_best=estimator.prior.t_index[estimator.i_best],
                z95_lo=estimator.z_limits[0], z95_hi=estimator.z_limits[-1],
                z68_lo=estimator.z_limits[1], z68_hi=estimator.z_limits[-2],
                z50=estimator.z_limits[2]
            ))
        except RuntimeError as e:
            num_failed += 1
            print('Estimator failed for i={}'.format(i))

        if ((print_interval and (i + 1) % print_interval == 0) or
            (i == num_batch - 1)):
            now = time.time()
            rate = (now - start_time) / (i + 1.)
            print('Completed {} / {} trials ({} failed) at {:.3f} sec/trial.'
                .format(i + 1, num_batch, num_failed, rate))

    return results
