# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run redshift estimator and analyze results.
"""
from __future__ import print_function, division

import time
import os

import numpy as np

import astropy.table
import astropy.constants
import astropy.units as u


def estimate_one(estimator, sampler, simulator, seed=1, i=0, mag_err=0.1,
                 save=None):
    """Run the estimator for a single simulated sample and display results.

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
    # TODO
    #BUG HERE USING sampler.name as the object type!!!!!!!!
    simulator.simulate(sampler.obs_wave, true_flux, sampler.name,
                       noise_generator=generator)
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
    """
    Run the estimator in batch mode and return a table of results that
    can be analyzed using plot_results(). Use the eval_bayes script to
    run batch estimates from the command line.

    Any individual fit can be studied in more detail by calling estimate_one
    with the same seed used here and the appropriate index value `i`.
    """
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
        # Use the name of the sampler as the type_name to the simulator.
        simulator.simulate(
            sampler.obs_wave, true_flux, sampler.name, noise_generator=generator)
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


def load_results(basename, seeds):
    """
    Load results produced by estimate_batch and saved as FITS files, e.g.
    by running the eval_bayes script.  Combines the results from multiple
    batch jobs run with different seeds into a single results table.

    If basename is a relative path then $BAYEZ_DATA will be pre-pended if it
    is defined.
    """
    results_path = os.environ.get('BAYEZ_DATA', '.')
    results = []
    for seed in seeds:
        name = os.path.join(results_path, '{}_{}.fits'.format(basename, seed))
        print('Loading {}'.format(name))
        results.append(astropy.table.Table.read(name))
    return astropy.table.vstack(results, join_type='exact')


def plot_slices(x, y, x_lo, x_hi, y_cut, num_slices=5, min_count=100):
    """
    Plotting utility function used by plot_results().  Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    # Assume that y is symmetric about zero.
    plt.xlim(x_lo, x_hi)
    plt.ylim(-1.25 * y_cut, +1.25 * y_cut)
    x_bins = np.linspace(x_lo, x_hi, num_slices + 1)
    x_i = np.digitize(x, x_bins) - 1
    limits = []
    counts = []
    for s in xrange(num_slices):
        y_slice = y[x_i == s]
        counts.append(len(y_slice))
        if counts[-1] > 0:
            limits.append(np.percentile(y_slice, (2.5, 16, 50, 84, 97.5)))
        else:
            limits.append((0., 0., 0., 0., 0.))
    limits = np.array(limits)
    counts = np.array(counts)

    # Plot scatter of all fits.
    plt.scatter(x, y, s=15, marker='.', lw=0, color='blue', alpha=0.5)

    # Plot quantiles in slices with enough fits.
    stepify = lambda y: np.vstack([y, y]).transpose().flatten()
    y_m2 = stepify(limits[:, 0])
    y_m1 = stepify(limits[:, 1])
    y_med = stepify(limits[:, 2])
    y_p1 = stepify(limits[:, 3])
    y_p2 = stepify(limits[:, 4])
    xstack = stepify(x_bins)[1:-1]
    for i in xrange(num_slices):
        s = slice(2 * i, 2 * i + 2)
        if counts[i] >= min_count:
            plt.fill_between(xstack[s], y_m2[s], y_p2[s], alpha=0.15, color='red')
            plt.fill_between(xstack[s], y_m1[s], y_p1[s], alpha=0.25, color='red')
            plt.plot(xstack[s], y_med[s], 'r-', lw=2.)

    # Plot cut lines.
    plt.axhline(+y_cut, ls='-', color='k')
    plt.axhline(-y_cut, ls='-', color='k')
    plt.grid()


def plot_results(results, z_min, z_max, mag_min, mag_max, mag_label, dv_cut,
                 num_z_slices=15, num_mag_slices=15, save=None):
    """
    Plot results obtained from estimate_batch() or load_results().
    Requires that matplotlib is installed.
    """
    import matplotlib.pyplot as plt

    CLIGHT_KM_S = astropy.constants.c.to(u.km / u.s).value
    #dv = CLIGHT_KM_S * results['dz_map'] / (1 + results['z'])
    dv = CLIGHT_KM_S * results['dz_avg'] / (1 + results['z'])
    #dv = CLIGHT_KM_S * (results['z50'] - results['z']) / (1 + results['z'])

    num_results = len(results)

    # Calculate the mean and std for the full sample.
    dv_mean = np.mean(dv)
    dv_stddev = np.std(dv)
    print('<dv> = {:.3f} km/s, std.dev. = {:.3f} km/s.'.format(dv_mean, dv_stddev))

    # Calculate quantiles for full sample.
    limits = np.percentile(dv, (2.5, 16, 50, 84, 97.5))
    # Error on the median from http://stats.stackexchange.com/questions/59838/standard-error-of-the-median
    median_error = 1.253 * dv_stddev / np.sqrt(num_results)
    print('median dv = {:.3f} +/- {:3f} km/s'.format(limits[2], median_error))
    print('68% interval = [{:+.3f},{:+.3f}] ~ +/- {:.3f} km/s'
          .format(limits[1], limits[-2], 0.5 * (limits[-2] - limits[1])))
    print('95% interval = [{:+.3f},{:+.3f}] ~ +/- {:.3f} km/s'
          .format(limits[0], limits[-1], 0.5 * (limits[-1] - limits[0])))

    # Count the number of 'catastrophic failures' with |dv| > 1000 km/s
    num_fail = np.count_nonzero(np.abs(dv) > 1000.)
    print('{} / {} = {:.2f}% spectra with |dv| > 1000 km/s.'
        .format(num_fail, num_results, 100. * num_fail / num_results))

    # Count the number of spectra matched to the correct simulation template.
    num_match = np.count_nonzero(results['t_true'] == results['t_best'])
    print('{} / {} = {:.2f}% spectra matched to the correct template.'
        .format(num_match, num_results, 100. * num_match / num_results))

    # Calculate coverage fractions.
    if 'z68_lo' in results.colnames:
        num_68 = np.count_nonzero(
            (results['z68_lo'] <= results['z']) & (results['z68_hi'] >= results['z']))
        num_95 = np.count_nonzero(
            (results['z95_lo'] <= results['z']) & (results['z95_hi'] >= results['z']))
        print('{:.2f}% ({:.2f}%) of estimates fall within 68% (95%) intervals.'
             .format(100. * num_68 / num_results, 100. * num_95 / num_results))
        num_50 = np.count_nonzero(results['z50'] <= results['z'])
        print('Median is below true z {:.2f}% of the time'
             .format(100. * num_50 / num_results))

    plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plot_slices(results['z'], dv, z_min, z_max, dv_cut, num_slices=num_z_slices)
    plt.xlabel('True redshift')
    plt.ylabel('Redshift error $\Delta v$ [km/s]')

    plt.subplot(1, 2, 2)
    plot_slices(results['mag'], dv, mag_min, mag_max, dv_cut, num_slices=num_mag_slices)
    plt.xlabel('True {} magnitude'.format(mag_label))
    plt.ylabel('Redshift error  $\Delta v$ [km/s]')

    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
