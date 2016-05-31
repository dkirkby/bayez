# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Support for redshift estimation.
"""
from __future__ import print_function, division

import numpy as np

import scipy.special
import scipy.interpolate

import numba

# Optimized inner loop for RedshiftEstimator.run()
@numba.jit(nopython=True)
def calculate_pull(ivar, flux, mi, mj, prior_flux, pulls):
    print(np.sum(prior_flux))
    print(np.sum(ivar))
    for k in range(mj.shape[0]): # magnitude abscissas
        flux_norm = 10 ** (-0.4 * (mj[k] - mi))
        for j in range(flux.shape[0]): # pixels j
            pulls[k, j] = ivar[j] * (flux[j] - flux_norm * prior_flux[j]) ** 2


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
        print("IN estimator.run(...)")
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
        print("BEFORE FOR LOOP")
        print(self.prior.flux.shape[0])
        for i in range(self.prior.flux.shape[0]):
            print("IN for loop: " + str(i))
            calculate_pull(
                ivar, flux, self.prior.mag[i], mj, self.prior.flux[i], pulls)

            # Sum over pixels to calculate chisq = -2 log(L) for the
            # spectroscopic likelihood L=P(D|m_j,i).
            chisq[i] = np.sum(pulls, axis=-1)



        print("After for loop")
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
