# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Simulate the instrument response in each camera band.

The following environment variables must be set to use this module.
Values below are for my laptop.

export DESIMODEL=/Users/david/Cosmo/DESI/code/desimodel
"""
from __future__ import print_function, division

import os

import numpy as np

import specsim

import desimodel.io


class Simulator(object):
    """
    Create a new instrument simulator.
    """

    def __init__(self, wavestep=0.2, instrument_downsampling=5,
                 analysis_downsampling=4, verbose=True):
        self.instrument_downsampling = instrument_downsampling
        self.analysis_downsampling = analysis_downsampling
        atmosphere = specsim.atmosphere.Atmosphere(
            skyConditions='dark', basePath=os.environ['DESIMODEL'])
        self.qsim = specsim.quick.Quick(
            atmosphere=atmosphere, basePath=os.environ['DESIMODEL'])
        # Configure the simulation the same way that quickbrick does so that our simulated
        # pixel grid matches the data challenge simulation pixel grid.
        desiparams = desimodel.io.load_desiparams()
        self.exptime = desiparams['exptime']
        if verbose:
            print('Exposure time is {}s.'.format(self.exptime))
        wavemin = desimodel.io.load_throughput('b').wavemin
        wavemax = desimodel.io.load_throughput('z').wavemax
        self.qsim.setWavelengthGrid(wavemin, wavemax, wavestep)
        if verbose:
            print('Simulation wavelength grid: ', self.qsim.wavelengthGrid)
        self.fluxunits = specsim.spectrum.SpectralFluxDensity.fiducialFluxUnit
        self.ranges = []
        self.num_analysis_pixels = 0
        # Pick the range of pixels to use from each camera in the analysis.
        for band in 'brz':
            j = self.qsim.instrument.cameraBands.index(band)
            R = self.qsim.cameras[j].sparseKernel
            resolution_limits = np.where(R.sum(axis=0).A[0] != 0)[0][[0,-1]]
            throughput_limits = np.where(
                self.qsim.cameras[j].throughput > 0)[0][[0,-1]]
            assert ((resolution_limits[0] < throughput_limits[0]) &
                    (resolution_limits[1] > throughput_limits[1])), \
                    'Unable to set band range.'
            if verbose:
                print('Band {}: simulation pixel limits are {}, {}.'.format(
                    band, throughput_limits[0], throughput_limits[1]))
            # Round limits to a multiple of the instrument downsampling so
            # that all simulation pixels in the included downsampling groups
            # have non-zero throughput.
            start = (throughput_limits[0] + instrument_downsampling - 1) // instrument_downsampling
            stop = throughput_limits[1] // instrument_downsampling
            # Trim the end of the range to give an even number of pixel groups
            # after analysis downsampling.
            band_analysis_pixels = (stop - start) // analysis_downsampling
            stop = start + band_analysis_pixels * analysis_downsampling
            if verbose:
                print('Band {}: downsampled aligned pixel limits are {}, {}.'
                    .format(band, start, stop))
            self.num_analysis_pixels += band_analysis_pixels
            self.ranges.append((start, stop))
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
        for band in 'brz':
            j = self.qsim.instrument.cameraBands.index(band)
            start, stop = self.ranges[j]
            n = (stop - start) // self.analysis_downsampling
            stop = start + n * self.analysis_downsampling
            # Average the flux over each analysis bin.
            instrument_flux = self.results['camflux'][start:stop, j]
            self.flux[base:base + n] = np.mean(
                instrument_flux.reshape(-1, self.analysis_downsampling), -1)
            # Sum the inverse variances over each analysis bin.
            instrument_ivar = self.results['camivar'][start:stop, j]
            self.ivar[base:base + n] = np.sum(
                instrument_ivar.reshape(-1, self.analysis_downsampling), -1)
            # Calculate the central wavelength of each analysis bin the first
            # time we are called.
            if self.wave is None:
                band_wave = self.results.wave[start:stop]
                wave[base:base + n] = np.mean(
                    band_wave.reshape(-1, self.analysis_downsampling), -1)
            base += n
        if self.wave is None:
            self.wave = np.copy(wave)
        assert np.all(self.ivar > 0), 'Some simulated pixels have ivar <= 0!'

    def simulate(self, wave, flux, airmass=1.25, noise_generator=None):
        """
        """
        inspec = specsim.spectrum.SpectralFluxDensity(
            wave, flux, fluxUnits=self.fluxunits, extrapolatedValue=True)
        self.results = self.qsim.simulate(
            sourceType='qso', sourceSpectrum=inspec,
            airmass=airmass, expTime=self.exptime,
            downsampling=self.instrument_downsampling)
        self.make_vectors()
        if noise_generator is not None:
            dflux = self.ivar ** -0.5
            self.flux += dflux * noise_generator.randn(self.num_analysis_pixels)

        return self.results
