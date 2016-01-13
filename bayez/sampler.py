# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Generate random samples of spectral templates.

The following environment variables must be set to use this module.
Values below are for my laptop.

export DESI_BASIS_TEMPLATES=/Data/DESI/basis_templates/v1.1
export DESISIM=/Users/david/Cosmo/DESI/code/desisim
#export DESIMODEL=/Users/david/Cosmo/DESI/code/desimodel
"""
from __future__ import print_function, division

import numpy as np
import scipy.special

import desisim.io
import desisim.filterfunc
import desisim.templates
import desisim.pixelsplines

import desimodel.io

import astropy.constants
import astropy.units as u
CLIGHT_KM_S = astropy.constants.c.to(u.km / u.s).value


class TemplateSampler(object):
    """Generic support for sampling spectroscopic templates.

    Subclasses handle the actual sampling for specific classes of objects.
    """
    def __init__(self, z_min, z_max, mag_min, mag_max, num_z_bins=50,
                 num_mag_bins=50, dwave=0.2):
        self.z_min, self.z_max = z_min, z_max
        self.mag_min, self.mag_max = mag_min, mag_max
        # Initialize our redshift and magnitude grids.
        self.z_bin_edges = np.linspace(z_min, z_max, num_z_bins + 1)
        self.z_grid = 0.5 * (self.z_bin_edges[:-1] + self.z_bin_edges[1:])
        self.mag_bin_edges = np.linspace(mag_min, mag_max, num_mag_bins + 1)
        self.mag_grid = 0.5 * (self.mag_bin_edges[:-1] + self.mag_bin_edges[1:])
        # Initialize the observed-frame wavelength grid to use.
        wave_min = desimodel.io.load_throughput('b').wavemin
        wave_max = desimodel.io.load_throughput('z').wavemax
        self.obs_wave = np.arange(wave_min - 1, wave_max + 1, dwave)

    def trim_templates(self):
        # Find the rest wavelength bounds required to cover all redshifts.
        wave_min = self.obs_wave[0] / (1 + self.z_max)
        wave_max = self.obs_wave[-1] / (1 + self.z_min)
        start = np.where(self.wave < wave_min)[0][-1]
        stop = np.where(self.wave > wave_max)[0][0] + 1
        print('trimming from {} to {} wavelength bins.'.format(
            len(self.wave),stop - start))
        self.wave = np.copy(self.wave[start:stop])
        self.spectra = np.copy(self.spectra[:, start:stop])

    def resample_flux(self, wave, flux):
        """
        Linearly interpolate the input spectrum to our observed wavelength grid.
        """
        interpolator = scipy.interpolate.interp1d(
            wave, flux, copy=False, assume_sorted=True, kind='linear')
        return interpolator(self.obs_wave)

    def plot_samples(self, num_samples=3, seed=None):
        """Plot some sample templates.

        This method requires that matplotlib is installed.
        """
        import matplotlib.pyplot as plt

        gen = np.random.RandomState(seed)
        plt.figure(figsize=(10, 4))
        for i in xrange(num_samples):
            spectrum, mag_pdf, z, mag, which = self.sample(generator=gen)
            label = '[{:04d}]z={:.2f},mag={:.1f}'.format(which, z, mag)
            plt.subplot(1, 2, 1)
            plt.plot(self.obs_wave, spectrum, label=label)
            plt.subplot(1, 2, 2)
            plt.plot(self.mag_grid, mag_pdf, label=label)
        plt.subplot(1, 2, 1)
        plt.xlabel('Wavelength $\lambda$ ($\AA$)')
        plt.ylabel('Template Flux $f(\lambda)$')
        plt.xlim(self.obs_wave[0], self.obs_wave[-1])
        plt.ylim(0., None)
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.legend(loc='lower left')
        plt.xlabel('Magnitude $m$')
        plt.ylabel('Probability Density $P(m|i)$')
        plt.xlim(self.mag_bin_edges[0], self.mag_bin_edges[-1])
        plt.ylim(0., None)
        plt.grid()
        plt.tight_layout()


class QSOSampler(TemplateSampler):
    """Sample QSO spectral templates."""

    def __init__(self, z_min=0.5, z_max=4.0, g_min=21., g_max=23.):
        TemplateSampler.__init__(self, z_min, z_max, g_min, g_max)
        # Load the template data
        spectra, self.wave, meta = desisim.io.read_basis_templates('QSO')
        keep = (meta['Z'] >= z_min) & (meta['Z'] <= z_max)
        self.num_templates = np.count_nonzero(keep)
        # Use flux units of 1e-17 * erg/cm/s/A
        self.spectra = 1e17 * spectra[keep]
        # Templates are already redshifted so we don't call trim_templates()
        self.template_z = meta['Z'][keep]
        # Calculate g-band magnitudes for each template.
        gfilter = desisim.filterfunc.filterfunc(filtername='decam_g.txt')
        self.gband = np.empty_like(self.template_z)
        for i, spectrum in enumerate(self.spectra):
            self.gband[i] = -2.5 * (np.log10(
                gfilter.get_maggies(self.wave, spectrum)) - 17.)

    def sample(self, generator=None):
        if generator is None:
            generator = np.random.RandomState()
        t_index = generator.choice(self.num_templates)
        # Lookup up this template's redshift.
        z = self.template_z[t_index]
        # Pick a g-band magnitude uniformly over the prior range.
        gmag = generator.uniform(self.mag_min, self.mag_max)
        gnorm = 10**(-0.4 * (gmag - self.gband[t_index]))
        # Resample to our observed wavelength grid.
        flux = self.resample_flux(self.wave, gnorm * self.spectra[t_index])
        # All magnitudes are equally likely.
        mag_pdf = np.ones_like(self.mag_grid)
        mag_pdf /= np.sum(mag_pdf)

        return flux, mag_pdf, z, gmag, t_index

def subdivide_normal(num_points, mean=0., sigma=1.):
    """Return points that sample a Gaussian with equal probability.
    """
    erf_edges = np.linspace(-1., +1., num_points + 1)
    erf_centers = 0.5 * (erf_edges[:-1] + erf_edges[1:])
    return mean + np.sqrt(2) * sigma * scipy.special.erfinv(erf_centers)

class LRGSampler(TemplateSampler):
    """Sample LRG spectral templates."""

    def __init__(self, z_min=0.5, z_max=1.1, zmag_min=19.0, zmag_max=20.5,
                 rmag_max=23., W1mag_max=19.35,
                 log10_vdisp_mean=2.3, log10_vdisp_rms=0.1, num_vdisp=5):
        TemplateSampler.__init__(self, z_min, z_max, zmag_min, zmag_max)
        # Load the template data
        self.spectra, self.wave, meta = desisim.io.read_basis_templates('LRG')
        self.num_templates = len(self.spectra)
        print('Loaded {} templates.'.format(self.num_templates))
        # Use flux units of 1e-17 * erg/cm/s/A
        self.spectra *= 1e17
        # Calculate magnitudes for each template.
        zfilter = desisim.filterfunc.filterfunc(filtername='decam_z.txt')
        rfilter = desisim.filterfunc.filterfunc(filtername='decam_r.txt')
        W1filter = desisim.filterfunc.filterfunc(filtername='wise_w1.txt')
        self.zband = np.empty((self.num_templates, len(self.z_grid)), np.float64)
        rband = np.empty_like(self.zband)
        W1band = np.empty_like(self.zband)
        for iz, z in enumerate(self.z_grid):
            wave = self.wave * (1 + z)
            for jt, spectrum in enumerate(self.spectra):
                self.zband[jt, iz] = -2.5 * (np.log10(
                    zfilter.get_maggies(wave, spectrum)) - 17.)
                rband[jt, iz] = -2.5 * (np.log10(
                    rfilter.get_maggies(wave, spectrum)) - 17.)
                W1band[jt, iz] = -2.5 * (np.log10(
                    W1filter.get_maggies(wave, spectrum)) - 17.)
        # Apply color-color cuts
        self.rz_color = rband - self.zband
        self.rW1_color = rband - W1band
        self.sel_color = (self.rz_color >= 1.6) & (self.rW1_color >= 1.3 * self.rz_color - 0.33)
        allowed = np.sum(self.sel_color, axis=-1)
        print('{} templates pass the color cuts.'
            .format(np.count_nonzero(allowed)))
        self.sel_indices = np.where(allowed)[0]
        # Calculate the z-band flux limits corresponding to the r-band and
        # W1-band cuts.
        self.zmag_max_rcut = rmag_max - self.rz_color
        self.zmag_max_W1cut = W1mag_max - (W1band - self.zband)
        self.zmag_max = np.minimum(self.zmag_max_rcut, self.zmag_max_W1cut)
        # Trim spectra after calculating magnitudes.
        self.trim_templates()
        # Calculate pixel boundaries to initialize Doppler broadening.
        pixbound = desisim.pixelsplines.cen2bound(self.wave)
        # Use a small set of fixed stellar velocity dispersions that sample a
        # Gaussian in log10(vdisp).
        vdisp_values = 10 ** subdivide_normal(
            num_vdisp, log10_vdisp_mean, log10_vdisp_rms)
        print('Using stellar velocity dispersions: {} km/s'
            .format(vdisp_values))
        self.blur_matrices = []
        for vdisp in vdisp_values:
            sigma = 1.0 + self.wave * vdisp / CLIGHT_KM_S
            self.blur_matrices.append(
                desisim.pixelsplines.gauss_blur_matrix(pixbound, sigma))

    def sample(self, generator=None):
        if generator is None:
            generator = np.random.RandomState()
        # Pick a random (template index, redshift) pair.
        selected = False
        while not selected:
            t_index = generator.choice(self.sel_indices)
            z_index = generator.choice(len(self.z_grid))
            # Pick a z-band magnitude uniformly the full prior range.
            zmag = generator.uniform(self.mag_min, self.mag_max)
            zmag_max = self.zmag_max[t_index, z_index]
            selected = self.sel_color[t_index, z_index] & (zmag <= zmag_max)

        # Randomize the redshift uniformly within this bin.
        z = generator.uniform(self.z_bin_edges[z_index],
                              self.z_bin_edges[z_index + 1])

        # Calculate the magnitude normalization factor.
        znorm = 10**(-0.4 * (zmag - self.zband[t_index, z_index]))
        # Pick a random velocity dispersion.
        vdisp_index = generator.choice(len(self.blur_matrices))
        blur = self.blur_matrices[vdisp_index]
        # Resample to our observed wavelength grid.
        flux = self.resample_flux(self.wave * (1 + z), znorm * (blur * self.spectra[t_index]))
        # All magnitudes up to zmag_max are equally likely.
        mag_pdf = np.ones_like(self.mag_grid)
        mag_pdf[self.mag_grid > zmag_max] = 0.
        mag_pdf /= np.sum(mag_pdf)

        return flux, mag_pdf, z, zmag, t_index

class ELGSampler(TemplateSampler):
    """Sample ELG spectral templates.

    For now, we only sample the EM continuum part of the template.
    """

    def __init__(self, z_min=0.6, z_max=1.6, r_min=21.0, r_max=23.4, foii_min=1.):
        TemplateSampler.__init__(self, z_min, z_max, r_min, r_max)#, num_z_bins=2)
        # Load the template data
        self.spectra, self.wave, meta = desisim.io.read_basis_templates('ELG')
        self.num_templates = len(self.spectra)
        print('Loaded {} templates.'.format(self.num_templates))
        self.d4000 = np.copy(meta['D4000'])
        self.ewoii = 10.0 ** (np.polyval([1.1074, -4.7338, 5.6585], self.d4000))
        self.oiiflux = meta['OII_CONTINUUM'] * self.ewoii
        # Use flux units of 1e-17 * erg/cm/s/A
        self.spectra *= 1e17
        self.oiiflux *= 1e17
        # Initialize bandpass filters
        gfilter = desisim.filterfunc.filterfunc(filtername='decam_g.txt')
        rfilter = desisim.filterfunc.filterfunc(filtername='decam_r.txt')
        zfilter = desisim.filterfunc.filterfunc(filtername='decam_z.txt')
        # Calculate the magnitudes of each normalized template on a grid of redshifts.
        self.rband = np.empty((self.num_templates, len(self.z_grid)), np.float64)
        gband = np.empty_like(self.rband)
        zband = np.empty_like(self.rband)
        for iz, z in enumerate(self.z_grid):
            wave = self.wave * (1 + z)
            for jt, spectrum in enumerate(self.spectra):
                self.rband[jt, iz] = -2.5 * (np.log10(
                    rfilter.get_maggies(wave, spectrum)) - 17.)
                gband[jt, iz] = -2.5 * (np.log10(
                    gfilter.get_maggies(wave, spectrum)) - 17.)
                zband[jt, iz] = -2.5 * (np.log10(
                    zfilter.get_maggies(wave, spectrum)) - 17.)

        # Trim spectra after calculating magnitudes.
        self.trim_templates()

        # Apply color-color cuts
        self.rz_color = self.rband - zband
        self.gr_color = gband - self.rband
        self.sel_color = ((self.rz_color >= 0.3) & (self.rz_color <= 1.5) &
                          (self.gr_color + 0.2 < self.rz_color) &
                          (self.rz_color < 1.2 - self.gr_color))
        allowed = np.sum(self.sel_color, axis=-1)
        print('{} templates pass color-color cuts.'
            .format(np.count_nonzero(allowed)))

        # Calculate maximum allowed rband magnitude to pass OII min flux cut.
        self.rmax_oii = self.rband + 2.5 * np.log10(
            self.oiiflux[:, np.newaxis] / foii_min)
        self.sel_oii = self.rmax_oii > self.mag_max
        allowed = np.sum(self.sel_oii, axis=-1)
        print('{} templates pass OII flux cut.'
            .format(np.count_nonzero(allowed)))

        self.sel_both = self.sel_oii & self.sel_color
        allowed = np.sum(self.sel_color & self.sel_oii, axis=-1)
        print('{} templates pass all cuts.'
            .format(np.count_nonzero(allowed)))
        self.sel_indices = np.where(allowed)[0]

    def sample(self, generator=None):
        """
        Return a randomly selected template with a redshift and r-band magnitude sampled
        from uniform priors, shaped by the color-color and OII minimum flux cuts.
        Each spectrum is resampled to a uniform observed wavelength grid.
        """
        if generator is None:
            generator = np.random.RandomState()
        # Pick a random (template index, redshift) pair.
        selected = False
        while not selected:
            t_index = generator.choice(self.sel_indices)
            z_index = generator.choice(len(self.z_grid))
            selected = self.sel_both[t_index, z_index]
        # Randomize the redshift uniformly within this bin.
        z = generator.uniform(self.z_bin_edges[z_index], self.z_bin_edges[z_index + 1])
        z_wave = self.wave * (1 + z)
        # Pick an r-band magnitude uniformly over the prior range.
        rmag = generator.uniform(self.mag_min, self.mag_max)
        rnorm = 10**(-0.4 * (rmag - self.rband[t_index, z_index]))
        # Resample to our observed wavelength grid.
        flux = self.resample_flux(self.wave * (1 + z), rnorm * self.spectra[t_index])
        # All magnitudes are equally likely.
        mag_pdf = np.ones_like(self.mag_grid)
        mag_pdf /= np.sum(mag_pdf)

        return flux, mag_pdf, z, rmag, t_index
