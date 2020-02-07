#! /usr/bin/env python
# Code to generate GRF mocks, in healpix format

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
import healpy as hp
import copy
import pymaster as nmt
from scipy.interpolate import interp1d
import sharedmem
import matplotlib.pyplot as plt

import compute_cl
reload(compute_cl)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Generate a suite of Gaussian mocks.')
parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)
parser.add_argument('--nproc', dest='nproc', type=int, help='Optional number of processes to run.', required=False)

### END OF PARSER ###



def enrich_params(params):
   """
   Infers the unspecified parameters from the parameters provided and
   updates the parameter dictionary accordingly.
   :param :
   :return :
   """
   
   params = copy.deepcopy(params)
   params['spins'] = np.array(params['spins'])
   # The ell array starts at 0, so the number of ells is lmax + 1
   if 'lmax' in params:
     params['nell'] = params['lmax']+1
   # coutning probes, fields, and their power spectra
   params['nprobes'] = len(params['probes'])
   params['ncls_probes'] = params['nprobes'] * (params['nprobes'] + 1) // 2
   params['nScalProbes'] = np.sum(params['spins']==0).astype('int')
   params['nTensProbes'] = np.sum(params['spins']==2).astype('int')
   params['nFields'] = params['nScalProbes'] + params['nTensProbes'] * 2
   params['ncls_fields'] = params['nFields'] * (params['nFields'] + 1) // 2
   return params




if __name__ == '__main__':

   args = parser.parse_args()

   # read parameters from yaml file
   logger.info('Read config from {}.'.format(args.path2config))
   config = yaml.load(open(args.path2config))
   # add inferred parameters from yaml file
   params = enrich_params(config)

   # create output directory if needed
   if not os.path.isdir(config['path2outputdir']):
      raise "Mock directory does not exist (gimme mocks)"

   # Only curved (healpix) maps for now
   if config['mode'] <> 'curved':
      raise NotImplementedError()

   # Number of processes to run on
   if config['nProc'] is None:
      nProc = 1
   else:
      nProc = config['nProc']
   logger.info('Running on '+str(nProc)+' processes')

   # generate and save all the mock maps
   #maps = generate_map(params, clArray, 0)
   nRea = config['nrealiz']
   with sharedmem.MapReduce(np=nProc) as pool:
      pool.map(analyzeMock, range(nRea))






def analyzeMock(iMock):
   

#!!!!! modify the newConfig for each pair of mocks, to give 
# the info needed for compute_cl
   newconfig = config.copy()
   newConfig['pathMap1'] = "" 

   compute_cl.compute(newConfig) 



