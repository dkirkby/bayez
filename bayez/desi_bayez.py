from __future__ import print_function, division
import numpy as np
import astropy.io.fits as fits
import astropy.table
import os.path
import specsim
import specsim.simulator

simulator = specsim.simulator.Simulator('DESI')
# Change
# atmosphere = specsim.atmosphere.Atmosphere(skyConditions='dark', basePath=os.environ['DESIMODEL'])
# qsim = specsim.quick.Quick(atmosphere=atmosphere, basePath=os.environ['DESIMODEL'])

# Configure the simulation the same way that quickbrick does so that our simulated
# pixel grid matches the data challenge simulation pixel grid.
# Get rid of this
# desiparams = desimodel.io.load_desiparams()
# exptime = desiparams['exptime']

# Not this
## read_native reads in native format fits files in order to avoid time in byte swapping
def read_native(hdus,name, dtype):
    assert np.dtype(dtype).isnative
    array = hdus[name].data.astype(dtype, casting='equiv', copy=True)
    del hdus[name].data
    return array
## read_brick reads DESI brick files, the object type can be qso, elg, star, lrg or mix
## it returns a list with 3 flux arrays, 3 ivar, and a numpy array with the ID list
def read_brick(objtype, path='', bricklist=[], verbose=True):
    """
    """
    wav = []
    fl =[]
    iv = []
    meta =[]
    for i, camera in enumerate(bricklist):
        try:
            xname = bricklist[i]
        except IndexError:
            print('brick list required!, format=[brickb,brickr,brickz] or [brickb,brickr] without extension')
        name = os.path.join(path, '{0}.fits'.format(bricklist[i]))
        if verbose:
            print('Reading {0}:'.format(name))
        hdulist = fits.open(name)
        nobj, nwave = hdulist[0].data.shape
        cwave = read_native(hdulist,'WAVELENGTH',np.float32)
        cobs = np.empty((nobj, nwave), dtype=[('flux', np.float32), ('ivar', np.float32)])
        cobs['flux'][:] = read_native(hdulist,'FLUX',np.float32)
        cobs['ivar'][:] = read_native(hdulist,'IVAR',np.float32)
        fl.append(cobs['flux']), iv.append(cobs['ivar']), wav.append(cwave)
        if i==0:
            idlist = hdulist[4].data['TARGETID']
    return  fl, iv, wav, idlist

def downsample(fl,iv,wav,analysis_downsampling=4):
    import desimodel.io
    import specsim.simulator
    simulator = specsim.simulator.Simulator(config)
    flux = np.empty((nobj,num_analysis_pixels), np.float32)
    ivar = np.empty((nobj,num_analysis_pixels), np.float32)
    wave = np.empty(num_analysis_pixels, np.float32)
    base=0
    for j, camera_output in enumerate(simulator.camera_output): #band in 'brz':
        n = simulator.band_sizes[j]
        # Average the flux over each analysis bin.
        instrument_wlen = camera_output['wavelength'] # self.results['camflux'][start:stop, j]
        end = -1 * (instrument_wlen.shape[0] % analysis_downsampling)
        if end is 0:
            end = None
        wstart = instrument_wlen[0]
        wstop = instrument_wlen[n]
        start_new = np.where(np.array(wav[j])>wstart)[0][0]
        stop_new = np.where(np.array(wav[j])>wstop)[0][0]

        n = (stop_new - start_new) // analysis_downsampling
        stop_new = start_new + n * analysis_downsampling
        # Average the flux over each analysis bin.
        instrument_flux = fl[j][:,start_new:stop_new]
        flux[:,base:base+n]= np.mean(instrument_flux.reshape(nobj,-1, analysis_downsampling), -1)
        # Sum the inverse variances over each analysis bin.
        instrument_ivar = iv[j][:,start_new:stop_new]
        ivar[:,base:base+n] = np.sum(instrument_ivar.reshape(nobj,-1, analysis_downsampling), -1)
        # Calculate the central wavelength of each analysis bin the first
        # time we are called.
        band_wave = wav[j][start_new:stop_new]
        wave[base:base+n] = np.mean(band_wave.reshape(-1, analysis_downsampling), -1)
        base += n
    return flux, ivar, wave

def estimate_desi(estimator,objtype,path='', bricklist=[]):
    fl, iv, wav,idlist = read_brick(objtype,path,bricklist)
    flux, ivar, wave = downsample(fl,iv,wav,analysis_downsampling=4)
    results = astropy.table.Table(
        names = ('i', 'z', 'p_best', 't_best', 'z95_lo', 'z68_lo', 'z50', 'z68_hi', 'z95_hi','z_best','zwarn','brickname','type','subtype'),
        dtype = ('i4', 'f4','i4', 'i4','f4', 'f4', 'f4', 'f4', 'f4','f4','i4','a8','a20','a20')
    )

    for i in range(0,flux.shape[0]):
        estimator.run(np.float32(flux[i]),np.float32(ivar[i]),-1,-1)
        results.add_row(dict(
            i=idlist[i],
            z=estimator.z_mean,
            p_best=estimator.i_best,
            t_best=estimator.prior.t_index[estimator.i_best],
            z95_lo=estimator.z_limits[0],
            z95_hi=estimator.z_limits[-1],
            z68_lo=estimator.z_limits[1],
            z68_hi=estimator.z_limits[-2],
            z50=estimator.z_limits[2],
            zwarn=0, #No warnings in this version
            brickname=bricklist[0],
            type=objtype,
            subtype='' #Not implemented in this version
        ))

    return results

def write_zbest(results, name='', path='', extrahdu=True):
    col1 = fits.Column(name='BRICKNAME',format='8A', array=results['brickname'])
    col2 = fits.Column(name='TARGETID', format='K', array=results['i'])
    col3 = fits.Column(name='Z', format='D', array=results['z'])
    col4 = fits.Column(name='ZERR',format='D', array=0.5*(results['z68_hi']-results['z68_lo']))
    col5 = fits.Column(name='ZWARN', format='K', array=results['zwarn'])
    col6 = fits.Column(name='TYPE', format='20A', array=results['type'])
    col7 = fits.Column(name='SUBTYPE', format='20A', array=results['subtype'])
    col8 = fits.Column(name='ZMAP', format='D', array=results['z_best'])
    col9 = fits.Column(name='Z50', format='D', array=results['z50'])
    col10 = fits.Column(name='Z95HI',format='D', array=results['z95_hi'])
    col11 = fits.Column(name='Z95LO',format='D', array=results['z95_lo'])
    col12 = fits.Column(name='Z68HI',format='D', array=results['z68_hi'])
    col13 = fits.Column(name='Z68LO',format='D', array=results['z68_lo'])
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7])
    cols2 = fits.ColDefs([col8,col9,col10,col11,col12,col13])
    tbhdu = fits.new_table(cols)
    tbhdu.name = 'ZBEST'
    tbhdu2 = fits.new_table(cols2)
    tbhdu2.name = 'BAYEZ'
    hdu = fits.PrimaryHDU(1)
    prihdr = fits.Header()
    prihdr['COMMENT']="Bayez redshift estimation. As for now error is computed as (zhi68-zlow68)/2"
    prihdu = fits.PrimaryHDU(header=prihdr)
    if(extrahdu): ##Extrahdu writes an extra hdu to include Bayez especific output
        thdulist = fits.HDUList([prihdu,tbhdu, tbhdu2])
    else:
        thdulist = fits.HDUList([prihdu,tbhdu])
    thdulist.writeto(path+name)
