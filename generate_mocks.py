#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Generate a suite of Gaussian mocks.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

### END OF PARSER ###

if __name__ == '__main__':

    args = parser.parse_args()

    config = yaml.load(open(args.path2config))
    logger.info('Read config from {}.'.format(args.path2config))

    nrealiz = config['simparams']['nrealiz']

    if config['mode'] == 'curved':
        from MockSurveyCurved_parallel import MockSurveyParallel
        import healpy as hp

        mask = hp.read_map(config['path2mask'])

    elif config['mode'] == 'flat':
        from MockSurvey_parallel import MockSurveyParallel
        from pixell import enmap

        mask = enmap.read_fits(config['path2mask'])

    else:
        raise NotImplementedError()

    # Here assuming for simplicity that masks are the same
    masks = [mask, mask, mask, mask, mask, mask]

    if 'l0_bins' in config['simparams']:
        config['simparams']['l0_bins'] = np.array(config['simparams']['l0_bins'])
    if 'l1_bins' in config['simparams']:
        config['simparams']['l1_bins'] = np.array(config['simparams']['l1_bins'])
    if 'spins' in config['simparams']:
        config['simparams']['spins'] = np.array(config['simparams']['spins'])

    if config['noiseparams'] is not None:
        logger.info('Generating noisy mocks.')
        mocksurvey = MockSurveyParallel(masks, config['simparams'], config['noiseparams'])
    else:
        logger.info('Generating noise-free mocks.')
        mocksurvey = MockSurveyParallel(masks, config['simparams'], noiseparams={})

    cls, noisecls, ells = mocksurvey.reconstruct_cls_parallel()

    if not os.path.isdir(config['path2outputdir']):
        try:
            os.makedirs(config['path2outputdir'])
        except:
            pass

    path2clarr = os.path.join(config['path2outputdir'], 'cls_signal-noise-removed_nrealis={}.npy'.format(nrealiz))
    np.save(path2clarr, cls)
    logger.info('Written signal cls to {}.'.format(path2clarr))

    path2clnoisearr = os.path.join(config['path2outputdir'], 'cls_noise_nrealis={}.npy'.format(nrealiz))
    np.save(path2clnoisearr, noisecls)
    logger.info('Written noise cls to {}.'.format(path2clnoisearr))

    path2ellarr = os.path.join(config['path2outputdir'], 'ells_uncoupled_nrealis={}.npy'.format(nrealiz))
    np.save(path2ellarr, ells)
    logger.info('Written ells to {}.'.format(path2ellarr))

    # for i in range(config['simparams']['nprobes']):
    #     for ii in range(i+1):
    #         path2wsp = os.path.join(config['path2outputdir'], 'wsp_probe1={}_probe2={}.dat'.format(i, ii))
    #         wsps[i][ii].write_to(path2wsp)