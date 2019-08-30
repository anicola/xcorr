#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from astropy.io import fits
import os
import pymaster as nmt
import copy
import healpy as hp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulatedMaps(object):
    """
    Class to generate maps and power spectra for a Gaussian
    mock joint survey including auto and cross correlations between
    all the probes.
    """

    def __init__(self,params={}):
        """
        Constructor for the SimulatedMaps class
        """

        self.params = params
        self.enrich_params()
        self.setup()
        self.print_params()

    def enrich_params(self):
        """
        Infers unspecified params from the specified ones and makes sure that all the facultative
        parameters are set.
        :return:
        """

        if 'tempbeam' not in self.params:
            self.params['tempbeam'] = False

    def print_params(self):
        """
        Prints the parameter combination chosen to initialise SimulatedMaps.
        """

        logger.info('SimulatedMaps has been initialised with the following attributes:')
        for key in self.params.keys():
            logger.info('{} = {}'.format(key, self.params[key]))

    def generate_maps(self):
        """
        Generates a set of maps by computing correlated realisations of the
        provided power spectra.
        :return:
        """

        logger.info('Generating Gaussian map realizations.')
        np.random.seed(seed=None)
        # Now create the maps with the correlations between both spin-0 and spin-2 fields


        print("tests")
        print("nside", self.params['nside'])
        print("cl shape", self.cls.shape)
        print("spins", self.params['spins'])
        print("pixwinarr shape", self.pixwinarr.shape)


        maps = nmt.synfast_spherical(self.params['nside'], self.cls, spin_arr=self.params['spins'], seed=-1, \
                                     beam=self.pixwinarr)
        logger.info('Gaussian maps done.')

        if self.params['nspin2'] > 0:
            logger.info('Spin 2 fields present. Reordering maps.')
            reordered_maps = self.reorder_maps(maps)

            if self.params['nspin2'] == 1:
                assert np.sum([np.all(maps[i] == reordered_maps[i]) for i in range(len(maps))]) == len(maps), \
                    'Something went wrong with map reordering.'
        else:
            logger.info('No spin 2 fields. Keeping map ordering.')
            reordered_maps = copy.deepcopy(maps)

        return reordered_maps

    def reorder_maps(self, maps):

        logger.info('Reordering maps.')

        tempmaps = copy.deepcopy(maps)

        spins = np.array(self.params['spins'])
        nspin2 = np.sum(spins == 2)
        ind = np.where(spins == 2)[0]
        min_ind = np.amin(ind)
        tempmaps[min_ind: min_ind+nspin2] = maps[min_ind::2]
        tempmaps[min_ind+nspin2:] = maps[min_ind+1::2]

        return tempmaps

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
                    cls[j+1, :] = np.zeros_like(cls_temp)
                    cls[j+2, :] = np.zeros_like(cls_temp)
                    j += 3
                elif self.params['spins'][i] == 2 and self.params['spins'][ii] == 0 or self.params['spins'][i] == 0 and self.params['spins'][ii] == 2:
                    cls[j+1, :] = np.zeros_like(cls_temp)
                    j += 2
                else:
                    j += 1

                k += 1

        return cls

    def setup(self):
        """
        Sets up derived parameters from the input parameters.
        :return:
        """

        logger.info('Setting up SimulatedMaps module.')

        # Read in the HEALPix pixel window function
        if self.params['pixwindow'] == 1:
            logger.info('pixwindow = {}.'.format(self.params['pixwindow']))
            logger.info('Applying pixel window function correction for NSIDE = {}.'.format(self.params['nside']))
#            PATH2HEALPIX = os.environ['HEALPIX']
#            hdu = fits.open(os.path.join(PATH2HEALPIX, 'data/pixel_window_n{}.fits'.format(str(self.params['nside']).zfill(4))))
#            pixwin = hdu[1].data['TEMPERATURE']
#            logger.info('Read {}.'.format(os.path.join(PATH2HEALPIX, 'data/pixel_window_n{}.fits'.format(str(self.params['nside']).zfill(4)))))
            pixwin = hp.sphtfunc.pixwin(self.params['nside'], pol=False, lmax=self.params['nell_theor']-1)
            self.pixwinarr = np.tile(pixwin, self.params['nprobes']).reshape((self.params['nprobes'], -1))

        else:
            logger.info('Pixel window function not supplied.')
            self.pixwinarr = None

        # Save the cls as a class attribute
        self.cls = self.read_cls()

        logger.info('Setup done!')










