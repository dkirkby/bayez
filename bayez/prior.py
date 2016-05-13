# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Support for redshift estimator priors.
"""
from __future__ import print_function, division

import numpy as np

import astropy.io.fits as fits

class Prior(object):

    def __init__(self, name):
        self.name = name

    def save(self, filename, clobber=False):
        # Create an empty primary HDU for header keywords
        primary = fits.PrimaryHDU()
        hdr = primary.header
        hdr['NAME'] = self.name
        hdr['Z_MIN'], hdr['Z_MAX'] = self.z_min, self.z_max
        hdr['MAG_MIN'], hdr['MAG_MAX'] = self.mag_min, self.mag_max
        # Save our internal arrays to ImageHDUs.
        wave = fits.ImageHDU(name='WAVE', data=self.wave)
        mag_grid = fits.ImageHDU(name='MAG_GRID', data=self.mag_grid)
        flux = fits.ImageHDU(name='FLUX', data=self.flux)
        mag_pdf = fits.ImageHDU(name='MAG_PDF', data=self.mag_pdf)
        z = fits.ImageHDU(name='Z', data=self.z)
        mag = fits.ImageHDU(name='MAG', data=self.mag)
        t_index = fits.ImageHDU(name='T_INDEX', data=self.t_index)
        # Write the file.
        hdus = fits.HDUList(
            [primary, wave, mag_grid, flux, mag_pdf, z, mag, t_index])
        hdus.writeto(filename, clobber=clobber)

    def plot(self, p_index=0):
        """Plot one spectrum from this prior.

        This method requires that matplotlib is installed.
        """
        import matplotlib.pyplot as plt
        label = '{}[{},{}] z={:.3f}, mag={:.2f}'.format(
            self.name, p_index, self.t_index[p_index], self.z[p_index], self.mag[p_index])
        plt.title(label)
        # Find the breakpoints between bands.
        r_start, z_start = np.where(np.diff(self.wave) < 0)[0] + 1
        # Plot each band with transparent overlaps.
        plt.fill_between(self.wave[:r_start], self.flux[p_index, :r_start],
                         color='b', alpha=0.25)
        plt.fill_between(self.wave[r_start:z_start],
                         self.flux[p_index, r_start:z_start],
                         color='r', alpha=0.25)
        plt.fill_between(self.wave[z_start:], self.flux[p_index, z_start:],
                         color='k', alpha=0.25)
        plt.xlim(self.wave[0], self.wave[-1])
        plt.ylim(0., None)
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux')
        plt.tight_layout()


def build_prior(name, sampler, simulator, num_priors,
                seed=None, print_interval=1000):

    # Allocate a new prior object.
    prior = Prior(name)
    prior.flux = np.empty(
        (num_priors, simulator.num_analysis_pixels), dtype=np.float32)
    prior.mag_pdf = np.empty(
        (num_priors, len(sampler.mag_grid)), dtype=np.float32)
    prior.z = np.empty((num_priors,), dtype=np.float32)
    prior.mag = np.empty((num_priors,), dtype=np.float32)
    prior.t_index = np.empty((num_priors,), dtype=np.int32)

    # Simulate and save the requested number of prior spectra.
    generator = np.random.RandomState(seed)
    for i in xrange(num_priors):
        spectrum, mag_pdf, z, mag, t_index = sampler.sample(generator)
        simulator.simulate(sampler.obs_wave, spectrum, name)
        prior.flux[i] = simulator.flux
        prior.mag_pdf[i] = mag_pdf
        prior.z[i] = z
        prior.mag[i] = mag
        prior.t_index[i] = t_index
        if print_interval and (i + 1) % print_interval == 0:
            print('Built {} / {} prior spectra.'.format(i + 1, num_priors))

    # Copy the simulation analysis wavelength bin centers.
    # Note that these are not valid until we have simulated at least spectrum.
    prior.wave = np.copy(simulator.wave)

    # Save the magnitude grid for renormalizing during marginalization.
    prior.mag_grid = np.copy(sampler.mag_grid)

    # Save the redshift and magnitude limits of this prior from the sampler.
    prior.z_min, prior.z_max = sampler.z_min, sampler.z_max
    prior.mag_min, prior.mag_max = sampler.mag_min, sampler.mag_max

    return prior


def load_prior(filename):
    hdus = fits.open(filename, memmap=False)
    hdr = hdus[0].header
    prior = Prior(name=hdr['NAME'])
    prior.z_min, prior.z_max = float(hdr['Z_MIN']), float(hdr['Z_MAX'])
    prior.mag_min, prior.mag_max = float(hdr['MAG_MIN']), float(hdr['MAG_MAX'])
    # Arrays in FITS files have dtypes like '>f4' where the '>' indicates
    # that bytes are in big-endian (aka network) order.  Since this is not
    # the native byte order on Intel platforms, perform byteswapping here.
    def read_native(name, dtype):
        assert np.dtype(dtype).isnative
        array = hdus[name].data.astype(dtype, casting='equiv', copy=True)
        del hdus[name].data
        return array
    prior.wave = read_native('WAVE', np.float32)
    prior.mag_grid = read_native('MAG_GRID', np.float64)
    prior.flux = read_native('FLUX', np.float32)
    prior.mag_pdf = read_native('MAG_PDF', np.float32)
    prior.z = read_native('Z', np.float32)
    prior.mag = read_native('MAG', np.float32)
    prior.t_index = read_native('T_INDEX', np.int32)
    hdus.close()
    return prior
