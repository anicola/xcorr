#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
import healpy as hp
import copy
import pymaster as nmt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Generate a suite of Gaussian mocks.')
parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

### END OF PARSER ###


def read_cl(params):
   """
   Reads in all the auto and cross power spectra needed to construct the set of
   correlated maps.
   It also multiplies the theoretical power spectra by the HEALPix pixel window
   functions and if the flag tempbeam is set it also multiplies them by the CMB
   beam window function.
   This is in order to test the pixel and beam window deconvolutions.
   :param :
   :return cls: 3D array with 0. and 1. axis denoting the number of the power spectrum and the
   3. axis is the power spectrum belonging to this index
   """

   logger.info('Setting up cl array.')
   #nspectra = params['ncls'] + params['nspin2']*(1+params['nprobes'])
   
   cls = np.zeros((params['ncls_full'], params['lmax']))
   print(cls.shape)
   logger.info('Cl array shape = {}.'.format(cls.shape))

   k = 0
   j = 0
   for i, probe1 in enumerate(params['probes']):
      for ii in range(i, params['nprobes']):

         probe2 = params['probes'][ii]
         logger.info('Reading cls for probe1 = {} and probe2 = {}.'.format(probe1, probe2))

         path2cls = params['path2cls'][k]
         data = np.genfromtxt(path2cls)
         logger.info('Read {}.'.format(path2cls))
         cls_temp = data[:params['lmax'], 1]
         # !!! check the size, and interpolate if needed

         cls[j, :] = cls_temp
         if params['spins'][i] == 2 and params['spins'][ii] == 2:
            #raise NotImplementedError()
            cls[j+1, :] = np.zeros_like(cls_temp)
            cls[j+2, :] = np.zeros_like(cls_temp)
            j += 3
         elif params['spins'][i] == 2 and params['spins'][ii] == 0 or params['spins'][i] == 0 and params['spins'][ii] == 2:
            #raise NotImplementedError()
            cls[j+1, :] = np.zeros_like(cls_temp)
            j += 2
         else:
            j += 1

         k += 1

   return cls





def enrich_params(params):
   """
   Infers the unspecified parameters from the parameters provided and
   updates the parameter dictionary accordingly.
   :param :
   :return :
   """
   
   params = copy.deepcopy(params)
   params['spins'] = np.array(params['spins'])
   params['nprobes'] = len(params['probes'])
   params['ncls_input'] = params['nprobes'] * (params['nprobes'] + 1) // 2
   if 'lmax' in params:
     params['nell'] = params['lmax']+1
   elif 'l0_bins'in params:
     params['nell'] = int(params['l0_bins'].shape[0])
   params['nspin2'] = np.sum(params['spins'] == 2).astype('int')
   params['nautocls'] = params['nprobes']+params['nspin2']
   params['nhppix'] = hp.nside2npix(params['nside'])
   params['nScalProbes'] = np.sum(params['spins']==0)
   params['nTensProbes'] = np.sum(params['spins']==2)
   params['nFields'] = params['nScalProbes'] + params['nTensProbes'] * 2
   params['ncls_full'] = params['nFields'] * (params['nFields'] + 1) // 2
   return params








def generate_map(params, cls, iRea):
   """
   Convenience method for calculating the signal and noise cls for
   a given mock realization. This is a function that can be pickled and can be thus
   used when running the mock generation in parallel using multiprocessing pool.
   :param realis: number of the realisation to run
   :param noise: boolean flag indicating if noise is added to the mocks
   noise=True: add noise to the mocks
   noise=False: do not add noise to the mocks
   :param probes: list of desired probes to run the mock for
   :param maskmat: matrix with the relevant masks for the probes
   :param clparams: list of dictionaries with the parameters for calculating the
   power spectra for each probe
   :return cls: 3D array of signal and noise cls for the given realisation,
   0. and 1. axis denote the power spectrum, 2. axis gives the cls belonging
   to this configuration
   :return noisecls: 3D array of noise cls for the given realisation,
   0. and 1. axis denote the power spectrum, 2. axis gives the cls belonging
   to this configuration
   :return tempells: array of the ell range of the power spectra
   """

   logger.info('Generate GRF realization: {}'.format(iRea))

   # Generate the GRF maps
   maps = nmt.synfast_spherical(params['nside'], cls, spin_arr=params['spins'], seed=iRea, \
         beam=None)

#   if params['nspin2'] > 0:
#      raise NotImplementedError()
#      logger.info('Spin 2 fields present. Reordering maps.')
#      reordered_maps = reorder_maps(maps)
#
#      if params['nspin2'] == 1:
#         assert np.sum([np.all(maps[i] == reordered_maps[i]) for i in range(len(maps))]) == len(maps), \
#               'Something went wrong with map reordering.'
#   else:
#      logger.info('No spin 2 fields. Keeping map ordering.')
#   reordered_maps = copy.deepcopy(maps)



   # Assuming all maps are either scalar or vector
#   nMaps = reordered_maps.shape[0]
#   nScalProbes = np.sum(params['spins']==0)
#   nTensProbes = np.sum(params['spins']==2)

   # Saving the scalar maps
   for iScalProbe in range(params['nScalProbes']):
      iMap = iScalProbe
      path = params['path2outputdir'] + '/' + params['probes'][iScalProbe] + '_nside' + str(params['nside']) + '_lmax' + str(params['lmax'])
      if params['mode'] == 'curved':
         path += '_curved'
      path += '_rea' + str(iRea) + '.fits'
      logger.info('Saving map to ' + path)
      hp.fitsfunc.write_map(path, maps[iMap], overwrite=True)


   # Saving the tensor maps
   iMap = params['nScalProbes']
   for iTensProbe in range(params['nScalProbes'], params['nprobes']):

      # save Q map as gamma 1
      path = params['path2outputdir'] + '/' + params['probes'][iTensProbe] + '_gamma1' + '_nside' + str(params['nside']) + '_lmax' + str(params['lmax'])
      if params['mode'] == 'curved':
         path += '_curved'
      path += '_rea' + str(iRea) + '.fits'
      logger.info('Saving map to ' + path)
      hp.fitsfunc.write_map(path, maps[iMap], overwrite=True)

      # save -U map as gamma 2
      path = params['path2outputdir'] + '/' + params['probes'][iTensProbe] + '_gamma2' + '_nside' + str(params['nside']) + '_lmax' + str(params['lmax'])
      if params['mode'] == 'curved':
         path += '_curved'
      path += '_rea' + str(iRea) + '.fits'
      logger.info('Saving map to ' + path)
      hp.fitsfunc.write_map(path, -maps[iMap+1], overwrite=True)
      
      # jump 2 by 2 for tensor maps
      iMap += 2


   return maps




#def reorder_maps( maps):
#   '''This seems to take a list of maps of the form
#   (T, T, T, Q, U, Q, U, Q, U)
#   and turn it into
#   (T, T, T, Q, Q, Q, U, U, U).
#   This assumes that the spin 0 fields are first,
#   followed by the spin 2 fields.
#   '''
#   raise NotImplementedError()
#
#   logger.info('Reordering maps.')
#
#   tempmaps = copy.deepcopy(maps)
#
#   spins = np.array(params['spins'])
#   nspin2 = np.sum(spins == 2)
#   ind = np.where(spins == 2)[0]
#   min_ind = np.amin(ind)
#   tempmaps[min_ind: min_ind+nspin2] = maps[min_ind::2]
#   tempmaps[min_ind+nspin2:] = maps[min_ind+1::2]
#
#   return tempmaps
#














if __name__ == '__main__':

   args = parser.parse_args()

   config = yaml.load(open(args.path2config))
   logger.info('Read config from {}.'.format(args.path2config))

   nrealiz = config['nrealiz']
   


   params = enrich_params(config)

   if config['mode'] == 'curved':
    pass
   else:
    raise NotImplementedError()

   if 'spins' in config:
    config['spins'] = np.array(config['spins'])

   if config['noisemodel'] is not None:
    logger.info('Generating noisy mocks.')
   else:
    logger.info('Generating noise-free mocks.')

   if not os.path.isdir(config['path2outputdir']):
    try:
       os.makedirs(config['path2outputdir'])
    except:
       pass

   # read the input theory cl
   # modify to add the noise power spectrum to the autos
   clArray = read_cl(params)


   # generate and save all the mock maps
   maps = generate_map(params, clArray, 0)
   # big parallel for loop
   # careful about the seeds
