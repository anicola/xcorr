#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import healpy as hp
import copy
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseMaps(object):
    """
    Class to generate noise-only maps to be used when simulating
    multiprobe surveys.
    """

    def __init__(self, noiseparams={}):
        """
        Constructor for the NoiseMaps class
        """

        self.params = noiseparams
        self.setup()
        self.print_params()

    def print_params(self):
        """
        Prints the parameter combination chosen to initialise NoiseMaps.
        """

        logger.info('NoiseMaps has been initialised with the following attributes:')
        for key in self.params.keys():
            print('{} = {}'.format(key, self.params[key]))

    def generate_maps(self, signalmaps=None):
        """
        Generates a list of noise-only maps for the defined probes.
        These are:
        CMB temperature: random permutation of the HMHD map
        galaxy overdensity: overdensity map generated by randomly placing
        Ngal galaxies in the survey footprint
        gamma: galaxy ellipticity map generated by rotating the shear of
         each galaxy by a random angle
        :param shearmap: 2D list with noisefree cosmic shear maps for e1 and e1 to
        add to the noisy ellipticities
        default: None
        :return maps: 1D list with map[0]=cmbnoisemap, map[1]=galaxy noise map,
        map[2]=ellipticity noise map
        """

        logger.info('Generating noise maps.')
        data = self.read_data()
        maps = [0 for i in range(self.params['nmaps'])]
        ii = 0
        for i, probe in enumerate(self.params['probes']):
            logger.info('Generating map for probe = {}.'.format(probe))
            if probe != 'gamma':
                if self.params['noisemodel'] == 'data':
                    maps[ii] = self.datanoisemap(probe, data[probe])
                else:
                    maps[ii] = self.gaussnoisemap(probe, data[probe])
                ii += 1
            else:
                if self.params['noisemodel'] == 'data':
                    if signalmaps is not None:
                        shearmaps = [signalmaps[i], signalmaps[i+self.params['nspin2']]]
                    else:
                        shearmaps = signalmaps

                    tempmaps = self.datanoisemap(probe, data[probe], shearmaps)
                else:
                    tempmaps = self.gaussnoisemap(probe, data[probe])
                maps[ii] = tempmaps[0]
                maps[ii+1] = tempmaps[1]
                ii += 2

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

        tempmaps = copy.deepcopy(maps)

        spins = np.array(self.params['spins'])
        nspin2 = np.sum(spins == 2)
        ind = np.where(spins == 2)[0]
        min_ind = np.amin(ind)
        tempmaps[min_ind: min_ind+nspin2] = maps[min_ind::2]
        tempmaps[min_ind+nspin2:] = maps[min_ind+1::2]

        return tempmaps

    def datanoisemap(self, probe, data, shearmaps=None):
        """
        Generates a noise-only HEALPix map for the specified probe using real data
        :param probe: string tag of desired probe
        :param data: the data needed to generate the noisemap
        CMB temperature & galaxy overdensity: HEALPix map
        gamma: structured array with galaxy position and ellipticity catalog
        :param shearmap: 2D list with noisefree cosmic shear maps for e1 and e1 to
        add to the noisy ellipticities
        default: None
        :return noisemap: HEALPix map of the noise for the respective probe
        """

        if probe == 'deltag':
            noisemap = self.randomize_deltag_map_fast(data)

        elif probe == 'gamma':
            # Add the cosmic shear to the noise ellipticites
            if shearmaps is not None:
                logger.info('Adding signal map to ellipticities.')
                pix_indices = hp.ang2pix(self.params['nside'], (90.0-(data['shearcat']['idec']))/180.0*np.pi, \
                                         data['shearcat']['ira']/180.0*np.pi, nest=False)

                data['shearcat']['ishape_hsm_regauss_e1_calib'] += shearmaps[0][pix_indices]
                data['shearcat']['ishape_hsm_regauss_e2_calib'] += shearmaps[1][pix_indices]

            noisemap = self.randomize_shear_map(data)

        else:
            raise NotImplementedError('Probes other than deltag and gamma not implemented yet. Aborting.')

        return noisemap

    def gaussnoisemap(self, probe, data):
        """
        Generates a noise-only HEALPix map as a Gaussian realisation of a theoretical noise power spectrum
        :param probe: string tag of desired probe
        :param data: the power spectrum needed to generate the noisemap
        :return noisemap: HEALPix map of the noise for the respective probe
        """

        if probe == 'gamma':
            # In the new healpy ordering the order of the power spectra is
            # TT, EE, BB, TE, EB, TB
            # and one needs to set at least 4 of those
            # We set the E and B mode power spectra to the theoretical shape noise power spectrum
            zeroarr = np.zeros(data['noisecls'].shape[0])
            ps = [zeroarr, data['noisecls'], data['noisecls'], zeroarr]
            tempmaps = list(hp.synfast(ps, self.params['nside'], new=True, pixwin=False))
            noisemap = [tempmaps[1], tempmaps[2]]
        else:
            noisemap = hp.synfast(data['noisecls'], self.params['nside'], new=True, pixwin=False)

        return noisemap

    def read_data(self):
        """
        Reads in the data needed for the generation of noise-only maps and
        saves them as class attributes.
        These are:
        CMB temperature: noise only map (half mission half difference map)
        galaxy overdensity: boolean mask of galaxy survey footprint
        gamma: structured array with galaxy position and ellipticity catalog
        :param:
        :return data: dictionary with data needed to generate noise realisations for each
        probe
        """

        data = {}
        for i, probe in enumerate(self.params['probes']):
            logger.info('Reading noise data for probe = {}.'.format(probe))
            logger.info('Noisemodel = {}.'.format(self.params['noisemodel']))

            if self.params['noisemodel'] == 'data':

                if probe == 'gamma':
                    # assert 'shearrot' in self.params, 'Requesting noise model from data but shearrot parameter not provided. Aborting.'
                    assert 'path2shearcat' in self.params, 'Requesting noise model from data but path2shearcat parameter not provided. Aborting.'

                    data[probe] = {}

                    hdulist = fits.open(self.params['path2shearcat'])
                    data[probe]['shearcat'] = hdulist[1].data
                    logger.info('Read {}.'.format(self.params['path2shearcat']))
                    if self.params['posfromshearcat'] == 0:
                        assert 'path2shearmask' in self.params, 'Requesting randomized galaxy positions for gamma but path2shearmask not provided. Aborting.'
                        tempmap = hp.read_map(self.params['path2shearmask'])
                        data[probe]['shearmask'] = tempmap

                elif probe == 'deltag':
                    assert 'Ngal' in self.params, 'Requesting noise model from data but Ngal parameter not provided. Aborting.'
                    assert 'path2deltagmask' in self.params, 'Requesting noise model from data but path to galaxy mask not provided. Aborting.'

                    data[probe] = {}

                    tempmap = hp.read_map(self.params['path2deltagmask'])
                    tempmask = tempmap.astype('bool').astype('int')
                    data[probe]['deltagmask'] = tempmask
                    logger.info('Read {}.'.format(self.params['path2deltagmask']))
                    data[probe]['Ngal'] = self.params['Ngal']

                else:
                    raise NotImplementedError('Probes other than deltag, gamma not implemented at the moment. Aborting.')

            else:
                assert 'path2noisecls' in self.params, 'Requesting theretical noise model but path2noisecls parameter not provided. Aborting.'

                data[probe] = {}

                data[probe]['noisecls'] = np.genfromtxt(self.params['path2noisecls'][i], usecols={1})
                logger.info('Read {}.'.format(self.params['path2noisecls']))

        return data

    def randomize_deltag_map_fast(self, data):
        """
        Creates a randomised version of the input map map by assigning the
        galaxies in the surevy to random pixels in the map. Basically it rotates each
        galaxy by a random angle but not rotating it out of the survey footprint.
        :param map: masked galaxy overdensity map which needs to randomised
        :param Ngal: number of galaxies used to create the map
        :return randomised_map: a randomised version of the masked input map
        """

        logger.info('Randomizing galaxy map.')

        mask = data['deltagmask']
        Ngal = data['Ngal']

        np.random.seed(seed=None)

        pixindices = np.where(mask == 1.)[0]
        galpix = np.random.choice(pixindices, size=Ngal)

        randomized_map = np.bincount(galpix, minlength=mask.shape[0]).astype('float')

        randomized_map = hp.ma(randomized_map)
        randomized_map.mask = copy.deepcopy(np.logical_not(mask))
        mean = np.mean(randomized_map)
        randomized_map = (randomized_map-mean)/mean

        return randomized_map

    def randomize_deltag_map(self, data):
        """
        Creates a randomised version of the input map map by assigning the
        galaxies in the surevy to random pixels in the map. Basically it rotates each
        galaxy by a random angle but not rotating it out of the survey footprint.
        :param map: masked galaxy overdensity map which needs to randomised
        :param Ngal: number of galaxies used to create the map
        :return randomised_map: a randomised version of the masked input map
        """

        logger.info('Randomizing galaxy map.')

        mask = data['deltagmask']
        Ngal = data['Ngal']

        np.random.seed(seed=None)

        pixindices = np.where(mask == 1.)[0]
        galpix = np.random.choice(pixindices, size=Ngal)

        randomized_map = np.zeros(mask.shape[0])
        for i in galpix:
            randomized_map[i] += 1.

        randomized_map = hp.ma(randomized_map)
        randomized_map.mask = copy.deepcopy(np.logical_not(mask))
        mean = np.mean(randomized_map)
        randomized_map = (randomized_map-mean)/mean

        return randomized_map

    def randomize_shear_map(self, data):

        logger.info('Randomizing shear map.')

        randomized_cat = self.randomize_shear_cat(data['shearcat'])
        if self.params['posfromshearcat'] == 1:
            e1map, e2map = self.make_curved_shear_map_fast(randomized_cat)
        else:
            e1map, e2map = self.make_curved_random_shear_map_fast(randomized_cat, data['shearmask'])

        return e1map, e2map

    def make_curved_shear_map_fast(self, cat):

        logger.info('Generating shear map based on catalog positions.')

        assert 'ishape_hsm_regauss_e1_calib' in cat.dtype.names, \
                                        'Shape catalog does not contain calibrated ellipticities. Aborting.'

        # Convert RA and DEC to HEALPix pixel numbers
        pix_indices = hp.ang2pix(self.params['nside'], (90.0-cat['idec'])/180.0*np.pi, cat['ira']/180.0*np.pi, nest=False)

        # Fill the HEALPix map
        N = hp.nside2npix(self.params['nside'])

        e1map = np.bincount(pix_indices, weights=cat['ishape_hsm_regauss_derived_shape_weight']*cat['ishape_hsm_regauss_e1_calib'], \
                            minlength=N)
        e2map = np.bincount(pix_indices, weights=cat['ishape_hsm_regauss_derived_shape_weight']*cat['ishape_hsm_regauss_e2_calib'], \
                            minlength=N)
        weightsmap = np.bincount(pix_indices, weights=cat['ishape_hsm_regauss_derived_shape_weight'], \
                            minlength=N)

        e1map[weightsmap != 0] /= weightsmap[weightsmap != 0]
        e2map[weightsmap != 0] /= weightsmap[weightsmap != 0]

        return e1map, e2map

    def make_curved_random_shear_map_fast(self, cat, mask):

        logger.info('Generating randomized position shear map.')

        assert 'ishape_hsm_regauss_e1_calib' in cat.dtype.names, \
                                        'Shape catalog does not contain calibrated ellipticities. Aborting.'

        maskpix = np.where(mask == 1.)[0]
        # Convert RA and DEC to HEALPix pixel numbers
        pix_indices = np.random.choice(maskpix, size=cat.shape[0])

        # Fill the HEALPix map
        N = hp.nside2npix(self.params['nside'])

        e1map = np.bincount(pix_indices, weights=cat['ishape_hsm_regauss_derived_shape_weight']*cat['ishape_hsm_regauss_e1_calib'], \
                            minlength=N)
        e2map = np.bincount(pix_indices, weights=cat['ishape_hsm_regauss_derived_shape_weight']*cat['ishape_hsm_regauss_e2_calib'], \
                            minlength=N)
        weightsmap = np.bincount(pix_indices, weights=cat['ishape_hsm_regauss_derived_shape_weight'], \
                            minlength=N)

        e1map[weightsmap != 0] /= weightsmap[weightsmap != 0]
        e2map[weightsmap != 0] /= weightsmap[weightsmap != 0]

        return e1map, e2map

    def make_curved_shear_map(self, cat):

        assert 'ishape_hsm_regauss_e1_calib' in cat.dtype.names, \
                                        'Shape catalog does not contain calibrated ellipticities. Aborting.'

        # Convert RA and DEC to HEALPix pixel numbers
        pix_indices = hp.ang2pix(self.params['nside'], (90.0-cat['idec'])/180.0*np.pi, cat['ira']/180.0*np.pi, nest=False)

        # Fill the HEALPix map
        N = hp.nside2npix(self.params['nside'])
        weightsmap = np.zeros(N)
        e1map = np.zeros(N)
        e2map = np.zeros(N)

        for i in range(len(pix_indices)):

            weightsmap[pix_indices[i]] += cat['ishape_hsm_regauss_derived_shape_weight'][i]
            e1map[pix_indices[i]] += cat['ishape_hsm_regauss_derived_shape_weight'][i]*cat['ishape_hsm_regauss_e1_calib'][i]
            e2map[pix_indices[i]] += cat['ishape_hsm_regauss_derived_shape_weight'][i]*cat['ishape_hsm_regauss_e2_calib'][i]

        e1map[weightsmap != 0] /= weightsmap[weightsmap != 0]
        e2map[weightsmap != 0] /= weightsmap[weightsmap != 0]

        return e1map, e2map

    def randomize_shear_cat(self, cat):
            """
            Rotates each galaxy ellipticity from the galaxy catalog data by a random angle to
            eliminate correlations between galaxy shapes.
            This is used to estimate the shape noise contribution to the shear power spectrum.
            :param cat: structured array with galaxy catalog to randomise
            :return randomiseddata: structured array with galaxy catalog with randomised ellipticities
            """

            logger.info('Randomizing shear catalogue.')

            assert 'ishape_hsm_regauss_e1_calib' in cat.dtype.names, \
            logger.warning('Catalog needs to contain calibrated shear columns.')

            # Copy the input data so it does not get overwritten
            randomized_cat = copy.deepcopy(cat)

            # Seed the random number generator
            np.random.seed(seed=None)

            thetarot = 2.*np.pi*np.random.random_sample((cat['ishape_hsm_regauss_e1_calib'].shape[0], ))

            randomized_cat['ishape_hsm_regauss_e1_calib'] = np.cos(2*thetarot)*cat['ishape_hsm_regauss_e1_calib'] - \
                                                     np.sin(2*thetarot)*cat['ishape_hsm_regauss_e2_calib']

            randomized_cat['ishape_hsm_regauss_e2_calib'] = np.sin(2*thetarot)*cat['ishape_hsm_regauss_e1_calib'] + \
                                                     np.cos(2*thetarot)*cat['ishape_hsm_regauss_e2_calib']

            return randomized_cat

    def setup(self):
        """
        Sets up derived parameters from the input parameters.
        :return:
        """

        logger.info('Setting up NoiseMaps module.')

        self.params['nmaps'] = len(self.params['probes']) + np.sum(self.params['spins'] == 2)

        logger.info('Setup done!')










