import numpy as np
import astropy.io.fits as fits
import astropy.table
import os.path
import specsim
import desimodel.io
import time
atmosphere = specsim.atmosphere.Atmosphere(skyConditions='dark', basePath=os.environ['DESIMODEL'])
qsim = specsim.quick.Quick(atmosphere=atmosphere, basePath=os.environ['DESIMODEL'])
# Configure the simulation the same way that quickbrick does so that our simulated
# pixel grid matches the data challenge simulation pixel grid.
desiparams = desimodel.io.load_desiparams()
exptime = desiparams['exptime']

## read_native reads in native format fits files in order to avoid time in byte swapping
def read_native(hdus,name, dtype):
    """
    Function to read natively fits files, prevents byte swapping and improves performance
    """
    assert np.dtype(dtype).isnative
    array = hdus[name].data.astype(dtype, casting='equiv', copy=True)
    del hdus[name].data
    return array
## read_brick reads DESI brick files, the object type can be qso, elg, star, lrg or mix
## it returns a list with 3 flux arrays, 3 ivar, and a numpy array with the ID list
def read_brick(objtype, path='', bricklist=[], verbose=True):
    """
    Function to read DESI brick files

    Arguments:

    objtype = Type of object 'elg','qso','star','lrg' are the types currently implemented
    path = Path of the brick files
    bricklist = List containing the name of the bricks (without the extension). Fits file format is assumed, [b,r,z] order is assumed
    """
    wav = []
    fl =[]
    iv = []
    meta =[]
    for i, camera in enumerate('brz'):
        try:
            xname = bricklist[i]
        except IndexError:
            print '3 brick files required!, format=[brickb,brickr,brickz] without extension'

        name = os.path.join(path, '{0}.fits'.format(bricklist[i]))
        if verbose:
            print 'Reading {0}:'.format(name)
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
def downsample(fl,iv,wav,analysis_downsampling=4, instrument_downsampling=5, wavestep=0.2):
    """
    Downsample the input from brick files (of 1A resolution) to accelerate the redshift estimation
    and match the pre-built priors' resolution (usually 2 or 4A).
    """
    import desimodel.io
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    qsim.setWavelengthGrid(wavemin, wavemax, wavestep)
    wav_temp = np.array(qsim.wavelengthGrid)
    nobj = len(fl[0])
    base=0
    num_analysis_pixels=0
    start=np.zeros(3)
    stop =np.zeros(3)
    for i,camera in enumerate('brz'):
        j = qsim.instrument.cameraBands.index(camera)
        throughput_limits = np.where(qsim.cameras[j].throughput > 0)[0][[0,-1]]
        # Round limits to a multiple of the instrument downsampling so
        # that all simulation pixels in the included downsampling groups
        # have non-zero throughput.
        start[i] = (throughput_limits[0] + instrument_downsampling - 1) // instrument_downsampling
        stop[i] = throughput_limits[1] // instrument_downsampling
        band_analysis_pixels = (stop[i] - start[i]) // analysis_downsampling
        num_analysis_pixels += band_analysis_pixels
    flux = np.empty((nobj,num_analysis_pixels), np.float32)
    ivar = np.empty((nobj,num_analysis_pixels), np.float32)
    wave = np.empty(num_analysis_pixels, np.float32)
    for i,camera in enumerate('brz'):
        wstart = wav_temp[start[i]*instrument_downsampling]
        wstop = wav_temp[stop[i]*instrument_downsampling]
        start_new = np.where(np.array(wav[i])>wstart)[0][0]
        stop_new = np.where(np.array(wav[i])>wstop)[0][0]

        n = (stop_new - start_new) // analysis_downsampling
        stop_new = start_new + n * analysis_downsampling
        # Average the flux over each analysis bin.
        instrument_flux = fl[i][:,start_new:stop_new]
        flux[:,base:base+n]= np.mean(instrument_flux.reshape(nobj,-1, analysis_downsampling), -1)
        # Sum the inverse variances over each analysis bin.
        instrument_ivar = iv[i][:,start_new:stop_new]
        ivar[:,base:base+n] = np.sum(instrument_ivar.reshape(nobj,-1, analysis_downsampling), -1)
        # Calculate the central wavelength of each analysis bin the first
        # time we are called.
        band_wave = wav[i][start_new:stop_new]
        wave[base:base+n] = np.mean(band_wave.reshape(-1, analysis_downsampling), -1)
        base += n
    return flux, ivar, wave

def estimate_desi(estimator,objtype,path='', bricklist=[]):
    """
    Function to estimate the redshift from a list of brick files
    Arguments:
    estimator = bayez.estimator object
    objtype = Type of object, 'elg', 'lrg', 'star' and 'qso' are implemented
    bricklist = list with the brick file names (without extension). b, r, z order is assumed
    """
    fl, iv, wav,idlist = read_brick(objtype,path,bricklist)
    flux, ivar, wave = downsample(fl,iv,wav,analysis_downsampling=4,instrument_downsampling=5, wavestep=0.2)
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
    """
    Function to write a DESI ZBEST file. If extrahdu=True it writes an extra hdu in the output
    to include some Bayez especific outputs
    """
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

def write_bricks(num_batch, classname, outname, path='', seed=1, mag_err=0.1, quadrature_order=16, print_interval=500, downsampling=4, verbose=True):
    """
    Run the estimator in batch mode and return a DESI brick file
    Any individual fit can be studied in more detail by calling estimate_one
    with the same seed used here and the appropriate index value `i`.
    """
    start_time = time.time()
    print('Starting at {}.'.format(time.ctime(start_time)))
    sampler = bayez.sampler.Samplers[classname]()
    simulator = bayez.simulation.Simulator(analysis_downsampling=downsampling, verbose=verbose)
    flux_arr = np.zeros((num_batch,len(simulator.flux)),dtype='float64')
    ivar_arr = np.zeros((num_batch,len(simulator.ivar)),dtype='float64')
    objtype=np.empty_like(np.zeros(num_batch),dtype='a10')
    targetcat=np.empty_like(objtype,dtype='a20')
    targetid=np.zeros(num_batch,dtype='int64')
    target_mask0=np.zeros(num_batch,dtype='int64')
    mag = np.zeros((num_batch,5),dtype='float32')
    FILTER = np.empty_like(objtype,dtype='a50')
    spectroid=np.empty_like(objtype,dtype='int64')
    positioner=np.zeros(num_batch,dtype='int64')
    fiber = np.random.randint(0,high=5000,size=num_batch)
    lambdaref=np.zeros(num_batch,dtype='float32')
    ra_target=np.zeros(num_batch,dtype='float64')
    dec_target=np.zeros(num_batch,dtype='float64')
    ra_obs=np.zeros(num_batch,dtype='float64')
    dec_obs=np.zeros(num_batch,dtype='float64')
    x_target=np.zeros(num_batch,dtype='float64')
    y_target=np.zeros(num_batch,dtype='float64')
    x_fvcobs=np.zeros(num_batch,dtype='float64')
    y_fvcobs=np.zeros(num_batch,dtype='float64')
    y_fvcerr=np.zeros(num_batch,dtype='float32')
    x_fvcerr=np.zeros(num_batch,dtype='float32')
    night = np.zeros(num_batch,dtype='int32')
    expid=np.zeros(num_batch,dtype='int32')
    index = np.empty_like(objtype,dtype='int32')
    for i in xrange(num_batch):
        generator = np.random.RandomState((seed, i))
        true_flux, mag_pdf, true_z, true_mag, t_index = (
            sampler.sample(generator))
        simulator.simulate(
            sampler.obs_wave, true_flux, noise_generator=generator)
        if(i==0): wav_arr=simulator.wave
        mag_obs = true_mag + mag_err * generator.randn()
        flux_arr[i]=simulator.flux
        ivar_arr[i]=simulator.ivar
        objtype[i]=classname
        targetcat[i]=classname
        targetid[i]=i
        index[i]=i
        if ((print_interval and (i + 1) % print_interval == 0) or
            (i == num_batch - 1)):
            now = time.time()
            rate = (now - start_time) / (i + 1.)
            print('Completed {} / {} trials at {:.3f} sec/trial.'
                .format(i + 1, num_batch, rate))
    print wav_arr
    head0 = fits.Header()
    head0.append(card=('NAXIS1',len(simulator.flux),'Number of wavelength bins'))
    head0.append(card=('NAXIS2',num_batch,'Number of spectra'))
    head0.append(card=('EXTNAME','FLUX','erg/s/cm^2/Angstrom'))
    hdu2 = fits.ImageHDU(data=wav_arr,name='WAVELENGTH')
    hdu3 = fits.ImageHDU(data=wav_arr,name='RESOLUTION')
    c1 = fits.Column(name='OBJTYPE', format='10A', array=objtype)
    c2 = fits.Column(name='TARGETCAT',format='20A', array=targetcat)
    c3 = fits.Column(name='TARGETID',format='K',array=targetid)
    c4 = fits.Column(name='TARGET_MASK0',format='K',array=target_mask0)
    c5 = fits.Column(name='MAG',format='5D',array=mag)
    c6 = fits.Column(name='FILTER',format='50A',array=FILTER)
    c7 = fits.Column(name='SPECTROID',format='K',array=spectroid)
    c8 = fits.Column(name='POSITIONER',format='K',array=positioner)
    c9 = fits.Column(name='FIBER',format='J',array=fiber)
    c10 = fits.Column(name='LAMBDAREF',format='E',array=lambdaref)
    c11 = fits.Column(name='RA_TARGET',format='D',array=ra_target)
    c12 = fits.Column(name='DEC_TARGET',format='D',array=dec_target)
    c13 = fits.Column(name='X_TARGET',format='D',array=x_target)
    c14 = fits.Column(name='Y_TARGET',format='D',array=y_target)
    c15 = fits.Column(name='X_FVCOBS',format='D',array=x_fvcobs)
    c16 = fits.Column(name='Y_FVCOBS',format='D',array=y_fvcobs)
    c17 = fits.Column(name='Y_FVCERR',format='E',array=y_fvcerr)
    c18 = fits.Column(name='X_FVCERR',format='E',array=x_fvcerr)
    c19 = fits.Column(name='NIGHT',format='J',array=night)
    c20 = fits.Column(name='EXPID',format='J',array=expid)
    c21 = fits.Column(name='INDEX',format='J',array=index)
    results = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9,
    c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21])
    breakpoints = np.where((wav_arr[1:]-wav_arr[:-1])<0)[0]+1
    print breakpoints
    start_arr = np.append(0,breakpoints)
    stop_arr = np.append(breakpoints,len(wav_arr))
    for j,band in enumerate('brz'):
        start=start_arr[j]
        stop=stop_arr[j]
        print start, stop
        hdu0=fits.PrimaryHDU(data=flux_arr[:,start:stop], header=head0)
        hdu1 = fits.ImageHDU(data=ivar_arr[:,start:stop],name='IVAR')
        hdulist = fits.HDUList([hdu0,hdu1,hdu2,hdu3,results])
        outfile = '%sbrick-%s-%s-%d.fits' %(path,band,outname,num_batch)
        hdulist.writeto(outfile,clobber=True)
