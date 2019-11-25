#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
import healpy as hp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Generate a suite of Gaussian mocks.')
parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

### END OF PARSER ###


    def read_cls(self):
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
        nspectra = self.params['ncls']+self.params['nspin2']+self.params['nspin2']*self.params['nprobes']
        cls = np.zeros((nspectra, self.params['nell_theor']))
        logger.info('Cl array shape = {}.'.format(cls.shape))

        k = 0
        j = 0
        for i, probe1 in enumerate(self.params['probes']):
            for ii in range(i, self.params['nprobes']):

                probe2 = self.params['probes'][ii]
                logger.info('Reading cls for probe1 = {} and probe2 = {}.'.format(probe1, probe2))

                path2cls = self.params['path2cls'][k]
                data = np.genfromtxt(path2cls)
                logger.info('Read {}.'.format(path2cls))
                cls_temp = data[:self.params['nell_theor'], 1]

                cls[j, :] = cls_temp
                if self.params['spins'][i] == 2 and self.params['spins'][ii] == 2:
                    raise NotImplementedError()
                    cls[j+1, :] = np.zeros_like(cls_temp)
                    cls[j+2, :] = np.zeros_like(cls_temp)
                    j += 3
                elif self.params['spins'][i] == 2 and self.params['spins'][ii] == 0 or self.params['spins'][i] == 0 and self.params['spins'][ii] == 2:
                    raise NotImplementedError()
                    cls[j+1, :] = np.zeros_like(cls_temp)
                    j += 2
                else:
                    j += 1

                k += 1

        return cls










def __call__(self, realis):
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

   logger.info('Running realisation: {}.'.format(realis))

   cls = np.zeros((self.params['nautocls'], self.params['nautocls'], self.params['nell']))
   noisecls = np.zeros_like(cls)

   logger.info('Generating Gaussian maps for one realization')
   np.random.seed(seed=None)
   # Now create the maps with the correlations between both spin-0 and spin-2 fields


   maps = nmt.synfast_spherical(self.params['nside'], self.cls, spin_arr=self.params['spins'], seed=-1, \
                               beam=self.pixwinarr)

   logger.info('Gaussian maps done for one realization')

   if self.params['nspin2'] > 0:
      raise NotImplementedError()
      logger.info('Spin 2 fields present. Reordering maps.')
      reordered_maps = self.reorder_maps(maps)

      if self.params['nspin2'] == 1:
          assert np.sum([np.all(maps[i] == reordered_maps[i]) for i in range(len(maps))]) == len(maps), \
              'Something went wrong with map reordering.'
   else:
      logger.info('No spin 2 fields. Keeping map ordering.')
      reordered_maps = copy.deepcopy(maps)

   # Save the maps to file here


   return




   def reorder_maps(self, maps):
     raise NotImplementedError()

     logger.info('Reordering maps.')

     tempmaps = copy.deepcopy(maps)

     spins = np.array(self.params['spins'])
     nspin2 = np.sum(spins == 2)
     ind = np.where(spins == 2)[0]
     min_ind = np.amin(ind)
     tempmaps[min_ind: min_ind+nspin2] = maps[min_ind::2]
     tempmaps[min_ind+nspin2:] = maps[min_ind+1::2]

     return tempmaps















if __name__ == '__main__':

    args = parser.parse_args()

    config = yaml.load(open(args.path2config))
    logger.info('Read config from {}.'.format(args.path2config))

    nrealiz = config['simparams']['nrealiz']

    if config['mode'] == 'curved':
        pass
    else:
        raise NotImplementedError()

    if 'spins' in config['simparams']:
        config['simparams']['spins'] = np.array(config['simparams']['spins'])

    if config['noiseparams'] is not None:
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
   clArray = read_cl()

   # generate and save all the mock maps
   # big parallel for loop
   # careful about the seeds
