# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Simulate the instrument response in each camera band.

The following environment variables must be set to use this module.
Values below are for my laptop::

    export DESIMODEL=/Users/david/Cosmo/DESI/code/desimodel
"""
from __future__ import print_function, division

import os

import numpy as np

import specsim

import specsim.simulator


class Simulator(object):
    """
    Create a new instrument simulator.

    Requires that desimodel be installed so that desimodel.io can be
    imported.
    """
    ### Input a config file get rid of wavestep and instrument_downsampling
    def __init__(self, config, analysis_downsampling=4, verbose=True):
        """
        config: A string or config object
        analysis_downsampling: Used to downsample from instrument output pixel
                               in order to work with smaller data vectors
        verbose: If true ouputs information regarding the details and status
                 of the simulation
        """
        self.analysis_downsampling = analysis_downsampling

        # Create the simulaton object
        self.simulator = specsim.simulator.Simulator(config)

        self.fluxunits = self.simulator.source.flux_in.unit #specsim uses erg/s/cm^{-2}/A as flux units
        self.waveunits = self.simulator.source.wavelength_in.unit
        self.band_sizes = []
        self.num_analysis_pixels = 0

        # Pick the range of pixels to use from each camera in the analysis.
        # Should be able to call wavelength_min/max on the camera objects
        for camera_output in self.simulator.camera_output:
            band_analysis_pixels = camera_output['observed_flux'].shape[0] // analysis_downsampling
            self.num_analysis_pixels += band_analysis_pixels
            self.band_sizes.append(band_analysis_pixels)

        if verbose:
            print('Total length of analysis pixel vector is {}.'
                .format(self.num_analysis_pixels))
        # Allocate vectors for data downsampled to analysis bins and flattened
        # over b,r,z.
        self.flux = np.empty((self.num_analysis_pixels,), np.float32)
        self.ivar = np.empty((self.num_analysis_pixels,), np.float32)
        # Will be allocated and filled in the first call to make_vectors.
        self.wave = None

    def make_vectors(self):
        base = 0
        if self.wave is None:
            wave = np.empty_like(self.flux)
        for j, camera_output in enumerate(self.simulator.camera_output): #band in 'brz':
            n = self.band_sizes[j]
            # Average the flux over each analysis bin.
            instrument_flux = camera_output['observed_flux'] # self.results['camflux'][start:stop, j]
            end = -1 * (instrument_flux.shape[0] % self.analysis_downsampling)
            if end is 0:
                end = None

            self.flux[base:base + n] = np.mean(
                instrument_flux[:end].reshape(-1, self.analysis_downsampling), -1)
            # Sum the inverse variances over each analysis bin.
            instrument_ivar = camera_output['flux_inverse_variance'] # self.results['camivar'][start:stop, j]
            self.ivar[base:base + n] = np.sum(
                instrument_ivar[:end].reshape(-1, self.analysis_downsampling), -1)
            # Calculate the central wavelength of each analysis bin the first
            # time we are called.
            if self.wave is None:
                band_wave = camera_output['wavelength'] # self.results.wave[start:stop]
                wave[base:base + n] = np.mean(
                    band_wave[:end].reshape(-1, self.analysis_downsampling), -1)
            base += n
        if self.wave is None:
            self.wave = np.copy(wave)
        assert np.all(self.ivar > 0), 'Some simulated pixels have ivar <= 0!'

    def simulate(self, wave, flux, type_name, noise_generator=None): # airmass=1.25, noise_generator=None):
        """
        """
        self.simulator.source.update_in(name="Not Meaningful", type_name=type_name, wavelength_in=(wave*self.waveunits), flux_in= (flux*self.fluxunits))
        self.simulator.source.update_out()

        self.simulator.simulate()

        self.make_vectors()

        ## Adding noise has been migrated into specsim
        if noise_generator is not None:
            dflux = self.ivar ** -0.5
            self.flux += dflux * noise_generator.randn(self.num_analysis_pixels)
