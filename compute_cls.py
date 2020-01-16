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

   # Only curved (healpix) maps for now
   if config['mode'] <> 'curved':
      raise NotImplementedError()

   if config['mode'] == 'curved':
      import healpy as hp

      spin1, spin2 = config['spins']
      logger.info('Spins: spin1 = {}, spin2 = {}.'.format(spin1, spin2))

      map1 = hp.read_map(config['pathMap1'], field=None)
      logger.info('Read map1 from {}.'.format(config['pathMap1']))
      nSide1 = hp.pixelfunc.get_nside(map1)
      map2 = hp.read_map(config['pathMap2'], field=None)
      logger.info('Read map2 from {}.'.format(config['pathMap2']))
      nSide2 = hp.pixelfunc.get_nside(map2)

      if nSide1<>nSide2:
         raise NotImplementedError()
      else:
         nSide = nSide1

      if config['pathMask1']=='None':
         mask1 = np.ones_like(map1)
      else:
         mask1 = hp.read_map(config['pathMask1'])
      logger.info('Read mask1 from {}.'.format(config['pathMask1']))
      
      
      if config['pathMask2']=='None':
         mask2 = np.ones_like(map2)
      else:
         mask2 = hp.read_map(config['pathMask2'])
      logger.info('Read mask2 from {}.'.format(config['pathMask2']))


      pixWin1, pixWin2 = config['pixWindow']
      if pixWin1==1:
         pixWin1 = hp.sphtfunc.pixwin(nSide1, pol=False)
      else:
         pixWin1 = None
      if pixWin2==1:
         pixWin2 = hp.sphtfunc.pixwin(nSide2, pol=False)
      else:
         pixWin2 = None


      if config['binSpacing']=='lin':
         lEdges = np.linspace(config['lMinCl'], config['lMaxCl'], config['nBinsCl']+1)
      if config['binSpacing']=='log':
         lEdges = np.logspace(np.log10(config['lMinCl']), np.log10(config['lMaxCl']), config['nBinsCl']+1)
      lEdges = lEdges.astype('int')
      

      # Generate the ell bins
      ells = np.arange(config['lMaxCl']+1)
      weights = np.zeros_like(ells, dtype='float64')
      bpws = -1*np.ones_like(ells)
      # Careful: bins start at zero
      for i in range(config['nBinsCl']):
         bpws[lEdges[i]:lEdges[i+1]] = i
         if config['binWeighting']=='uniform':
            weights[lEdges[i]:lEdges[i+1]] = 1.
         elif config['binWeighting']=='numModes':
            weights[lEdges[i]:lEdges[i+1]] = 2. * ells[lEdges[i]:lEdges[i+1]] + 1.
      b = nmt.NmtBin(nSide, bpws=bpws, ells=ells, weights=weights)
      # The effective sampling rate for these bandpowers can be obtained calling:
      ells_uncoupled = b.get_effective_ells()

      # Only spins 0 for now
      if spin1<>0 or spin2<>0:
         raise RaiseNotImplementedError()

#!!! this assumes that gamma1 and gamma2 are part of one healpix map,
# where the T field is empty. Hence indices 1 and 2 rather than 0 and 1
      if spin1==2 and spin2==0:
         raise NotImplementedError()
         emaps1 = [map1[1], map1[2]]
         emaps2 = [map2]
      elif spin1==0 and spin2==2:
         raise NotImplementedError()
         emaps1 = [map1]
         emaps2 = [map2[1], map2[2]]
      elif spin1==2 and spin2==2:
         raise NotImplementedError()
         emaps1 = [map1[1], map1[2]]
         emaps2 = [map2[1], map2[2]]
      else:
         emaps1 = [map1]
         emaps2 = [map2]

      # Define fields
      f1 = nmt.NmtField(mask1, emaps1, purify_b=False, beam=pixWin1)
      f2 = nmt.NmtField(mask2, emaps2, purify_b=False, beam=pixWin2)

      logger.info('Computing workspace element.')
      wsp = nmt.NmtWorkspace()
      wsp.compute_coupling_matrix(f1, f2, b)

      # Compute pseudo-Cls
      logger.info('Computing (coupled) pseudo-cls')
      cl_coupled = nmt.compute_coupled_cell(f1, f2)
      # Uncoupling pseudo-Cls
      # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
      # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
      logger.info('Decoupling cls.')
      cl_uncoupled = wsp.decouple_cell(cl_coupled)

      cl_out = np.vstack((ells_uncoupled, cl_uncoupled))

   pathOutputDir = config['pathOutputDir']
   if not os.path.isdir(pathOutputDir):
      try:
         os.makedirs(pathOutputDir)
         logger.info('Created directory {}.'.format(pathOutputDir))
      except:
         logger.info('Directory {} already exists.'.format(pathOutputDir))
         pass


   path = pathOutputDir + "/" + config['nameOutputClFile']
   np.savetxt(path, cl_out.T)
   logger.info('Written cls to {}.'.format(path))

   #wsp.write_to(config['path2wsp'])
   #logger.info('Written wsp to {}.'.format(config['path2wsp']))
