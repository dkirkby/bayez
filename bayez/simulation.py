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
    def __init__(self, config,  # wavestep=0.2, instrument_downsampling=5,
                 analysis_downsampling=4, verbose=True):
        """
        config: A string or config object
        analysis_downsampling: Used to downsample from instrument output pixel
                               in order to work with smaller data vectors
        verbose: If true ouputs information regarding the details and status
                 of the simulation
        """
        # This does not need to be imported anymore.
        # import desimodel.io

        # This instrument downsampling is now being handeld in specsim
        # self.instrument_downsampling = instrument_downsampling

        self.analysis_downsampling = analysis_downsampling


        # The creation of the atmosphere is now handeled within the simulator object
        #atmosphere = specsim.atmosphere.Atmosphere(
            #skyConditions='dark', basePath=os.environ['DESIMODEL'])

        # Create the simulaton object
        self.simulator = specsim.simulator.Simulator(config)

        # Quick sim no longer exists
        #self.qsim = specsim.quick.Quick(
            #atmosphere=atmosphere, basePath=os.environ['DESIMODEL'])

        # Configure the simulation the same way that quickbrick does so that our simulated
        # pixel grid matches the data challenge simulation pixel grid.
        # desiparams = desimodel.io.load_desiparams()

        # I'm not sure we have to bother getting this stuff anymore
        # we can probably delete most of this code.
        # self.exptime = self.simulator.instrument.exposure_time # desiparams['exptime']
        # if verbose:
        #     print('Exposure time is {}s.'.format(self.exptime))
        # wavemin = self.simulator.instrument.wavelength_min # desimodel.io.load_throughput('b').wavemin
        # wavemax = self.simulator.instrument.wavelength_max # desimodel.io.load_throughput('z').wavemax

        # self.qsim.setWavelengthGrid(wavemin, wavemax, wavestep)
        # if verbose:
        #     print('Simulation wavelength grid: ', self.qsim.wavelengthGrid)

        self.fluxunits = self.simulator.source.flux_in.unit # specsim.spectrum.SpectralFluxDensity.fiducialFluxUnit
        #self.ranges = []
        self.band_sizes = []
        self.num_analysis_pixels = 0

        # Pick the range of pixels to use from each camera in the analysis.
        # Should be able to call wavelength_min/max on the camera objects
        for camera in self.simulator.instrument.cameras: # for band in 'brz':
            # j = self.qsim.instrument.cameraBands.index(band)
            # R = self.qsim.cameras[j].sparseKernel
            # resolution_limits = np.where(R.sum(axis=0).A[0] != 0)[0][[0,-1]]
            # throughput_limits = np.where(
            #     self.qsim.cameras[j].throughput > 0)[0][[0,-1]]
            # assert ((resolution_limits[0] < throughput_limits[0]) &
            #         (resolution_limits[1] > throughput_limits[1])), \
            #         'Unable to set band range.'
            # if verbose:
            #     print('Band {}: simulation pixel limits are {}, {}.'.format(
            #         band, throughput_limits[0], throughput_limits[1]))
            # # Round limits to a multiple of the instrument downsampling so
            # # that all simulation pixels in the included downsampling groups
            # # have non-zero throughput.
            # start = (throughput_limits[0] + instrument_downsampling - 1) // instrument_downsampling
            # stop = throughput_limits[1] // instrument_downsampling
            # # Trim the end of the range to give an even number of pixel groups
            # # after analysis downsampling.

            # We could get rid of start and stop this and just use ccd_coverage (see below)
            # start = camera.ccd_slice.start
            # stop = cam_slice.stop
            # band_analysis_pixels = (stop - start) // analysis_downsampling
            # stop = start + band_analysis_pixels * analysis_downsampling
            # if verbose:
            #     print('Band {}: downsampled aligned pixel limits are {}, {}.'
            #         .format(band, start, stop))
            # self.num_analysis_pixels += band_analysis_pixels
            # self.ranges.append((start, stop))

            # OR could just do
            band_analysis_pixels = np.where(camera.ccd_coverage)[0].shape[0] // analysis_downsampling
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
        for j, camrea_output in enumerate(self.simulator.camera_output): #band in 'brz':
            #j = self.qsim.instrument.cameraBands.index(band)
            # Don't need start and stop anymore just use ccd_coverage (see above)
            # start, stop = self.ranges[j]
            # n = (stop - start) // self.analysis_downsampling
            # stop = start + n * self.analysis_downsampling

            #REPLACE WITH
            n = self.band_sizes[j]

            # Average the flux over each analysis bin.
            instrument_flux = camera_output['obserbed_flux'] # self.results['camflux'][start:stop, j]
            self.flux[base:base + n] = np.mean(
                instrument_flux.reshape(-1, self.analysis_downsampling), -1)
            # Sum the inverse variances over each analysis bin.
            instrument_ivar = camera_output['flux_inverse_variance'] # self.results['camivar'][start:stop, j]
            self.ivar[base:base + n] = np.sum(
                instrument_ivar.reshape(-1, self.analysis_downsampling), -1)
            # Calculate the central wavelength of each analysis bin the first
            # time we are called.
            if self.wave is None:
                band_wave = camera_output['wavelength'] # self.results.wave[start:stop]
                wave[base:base + n] = np.mean(
                    band_wave.reshape(-1, self.analysis_downsampling), -1)
            base += n
        if self.wave is None:
            self.wave = np.copy(wave)
        assert np.all(self.ivar > 0), 'Some simulated pixels have ivar <= 0!'

    def simulate(self, wave, flux, type_name, noise_generator): # airmass=1.25, noise_generator=None):
        """
        """
        #SpectralFluxDensity: The spectrum of the source without the sky
        # Change to update_in from source.py
        # inspec = specsim.spectrum.SpectralFluxDensity(
        #     wave, flux, fluxUnits=self.fluxunits, extrapolatedValue=True)
        # Not sure what to do about the name and type name parameters.
        # Should they be passed into the method
        self.simulator.source.update_in(name="Not Meaninful", type_name=type_name, wavelengn_in=wave, flux_in=flux)


        # self.results = self.qsim.simulate(
        #     sourceType='qso', sourceSpectrum=inspec,
        #     airmass=airmass, expTime=self.exptime,
        #     downsampling=self.instrument_downsampling)
        self.simulator.simulate()


        self.make_vectors()
        ## Adding noise has been migrated into specsim
        if noise_generator is not None:
            dflux = self.ivar ** -0.5
            self.flux += dflux * noise_generator.randn(self.num_analysis_pixels)

        # Should this method stil return something? No
        #return self.results
