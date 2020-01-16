#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
from astropy.io import fits
# Import the NaMaster python wrapper
import pymaster as nmt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Compute cls using NaMaster.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

### END OF PARSER ###

if __name__ == '__main__':

    args = parser.parse_args()

    config = yaml.load(open(args.path2config))
    logger.info('Read config from {}.'.format(args.path2config))

    if config['mode'] == 'curved':
        import healpy as hp

        spin1, spin2 = config['spins']
        logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

        map1 = hp.read_map(config['path2map1'], field=None)
        logger.info('Read map1 from {}.'.format(config['path2map1']))
        map2 = hp.read_map(config['path2map2'], field=None)
        logger.info('Read map2 from {}.'.format(config['path2map2']))

        mask1 = hp.read_map(config['path2mask1'])
        logger.info('Read mask1 from {}.'.format(config['path2mask1']))
        mask2 = hp.read_map(config['path2mask2'])
        logger.info('Read mask2 from {}.'.format(config['path2mask2']))

        if config['pixwindow'] == 1:
            logger.info('Applying pixel window function correction for NSIDE = {}.'.format(config['nside']))
            PATH2HEALPIX = os.environ['HEALPIX']
            hdu = fits.open(os.path.join(PATH2HEALPIX, 'data/pixel_window_n{}.fits'.format(config['nside'])))
            pixwin = hdu[1].data['TEMPERATURE']
            logger.info('Read {}.'.format(os.path.join(PATH2HEALPIX, 'data/pixel_window_n{}.fits'.format(config['nside']))))
            pixwin = pixwin[:3*config['nside']]
        else:
            logger.info('Not applying pixel window function correction for NSIDE = {}.'.format(config['nside']))
            pixwin = None

        l0_bins = np.around(config['l0_bins']).astype('int')
        lf_bins = np.around(config['lf_bins']).astype('int')

        ells = np.arange(np.amax(lf_bins))
        weights = np.zeros_like(ells, dtype='float64')
        bpws = -1*np.ones_like(ells)
        # Careful: bins start at zero
        for i in range(l0_bins.shape[0]):
            bpws[l0_bins[i]:lf_bins[i]] = i
            weights[l0_bins[i]:lf_bins[i]] = 1./(lf_bins[i]-l0_bins[i])

        b = nmt.NmtBin(config['nside'], bpws=bpws, ells=ells, weights=np.ones_like(ells))
        # The effective sampling rate for these bandpowers can be obtained calling:
        ells_uncoupled = b.get_effective_ells()

        if spin1 == 2 and spin2 == 0:
            # Define curved sky spin-2 map
            emaps = [map1[1], map1[2]]
            # Define curved sky spin-0 map
            emaps = [map2]

        elif spin1 == 0 and spin2 == 1:
            # Define curved sky spin-2 map
            emaps = [map1]
            # Define curved sky spin-0 map
            emaps = [map2[1], map2[2]]

        elif spin1 == 2 and spin2 == 2:
            # Define flat sky spin-2 map
            emaps = [map1[1], map1[2]]
            # Define flat sky spin-0 map
            emaps = [map2[1], map2[2]]

        else:
            # Define flat sky spin-0 map
            emaps = [map1]
            # Define flat sky spin-0 map
            emaps = [map2]

        # Define fields
        f1 = nmt.NmtField(mask1, emaps, purify_b=False, beam=pixwin)
        f2 = nmt.NmtField(mask2, emaps, purify_b=False, beam=pixwin)

        logger.info('Computing workspace element.')
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(f1, f2, b)

        # Compute pseudo-Cls
        cl_coupled = nmt.compute_coupled_cell(f1, f2)
        # Uncoupling pseudo-Cls
        # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
        # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
        logger.info('Decoupling cls.')
        cl_uncoupled = wsp.decouple_cell(cl_coupled)

        cl_out = np.vstack((ells_uncoupled, cl_uncoupled))

    elif config['mode'] == 'flat':
        from pixell import enmap

        spin1, spin2 = config['spins']
        logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

        map1 = enmap.read_fits(config['path2map1'])
        logger.info('Read map1 from {}.'.format(config['path2map1']))
        map2 = enmap.read_fits(config['path2map2'])
        logger.info('Read map2 from {}.'.format(config['path2map2']))

        mask1 = enmap.read_fits(config['path2mask1'])
        logger.info('Read mask1 from {}.'.format(config['path2mask1']))
        mask2 = enmap.read_fits(config['path2mask2'])
        logger.info('Read mask2 from {}.'.format(config['path2mask2']))

        b = nmt.NmtBinFlat(config['l0_bins'], config['lf_bins'])
        # The effective sampling rate for these bandpowers can be obtained calling:
        ells_uncoupled = b.get_effective_ells()

        if spin1 == 2 and spin2 == 0:
            # Define flat sky spin-2 map
            emaps = [map1[0], map1[1]]
            # Define flat sky spin-0 map
            emaps = [map2]

        elif spin1 == 2 and spin2 == 0:
            # Define flat sky spin-2 map
            emaps = [map1[0], map1[1]]
            # Define flat sky spin-0 map
            emaps = [map2]

        elif spin1 == 2 and spin2 == 2:
            # Define flat sky spin-2 map
            emaps = [map1[0], map1[1]]
            # Define flat sky spin-0 map
            emaps = [map2[0], map2[1]]

        else:
            # Define flat sky spin-0 map
            emaps = [map1]
            # Define flat sky spin-0 map
            emaps = [map2]

        # Define fields
        f1 = nmt.NmtFieldFlat(config['Lx'], config['Ly'], mask1, emaps, purify_b=False)
        f2 = nmt.NmtFieldFlat(config['Lx'], config['Ly'], mask2, emaps, purify_b=False)

        logger.info('Computing workspace element.')
        wsp = nmt.NmtWorkspaceFlat()
        wsp.compute_coupling_matrix(f1, f2, b)

        # Compute pseudo-Cls
        cl_coupled = nmt.compute_coupled_cell_flat(f1, f2, b)
        # Uncoupling pseudo-Cls
        # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
        # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
        logger.info('Decoupling cls.')
        cl_uncoupled = wsp.decouple_cell(cl_coupled)

        cl_out = np.vstack((ells_uncoupled, cl_uncoupled))

    else:
        raise NotImplementedError()

    path2outputdir, _ = os.path.split(config['path2cls'])
    if not os.path.isdir(path2outputdir):
        try:
            os.makedirs(path2outputdir)
            logger.info('Created directory {}.'.format(path2outputdir))
        except:
            logger.info('Directory {} already exists.'.format(path2outputdir))
            pass

    np.savetxt(config['path2cls'], cl_out.T)
    logger.info('Written cls to {}.'.format(config['path2cls']))

    wsp.write_to(config['path2wsp'])
    logger.info('Written wsp to {}.'.format(config['path2wsp']))
