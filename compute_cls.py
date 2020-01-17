#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import argparse
import logging
import numpy as np
import yaml
import os
from astropy.io import fits
from time import time
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
# Import the NaMaster python wrapper
import pymaster as nmt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### START OF PARSER ###

parser = argparse.ArgumentParser(description='Compute cls using NaMaster.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to yaml config file.', required=True)

parser.add_argument('--pathOutputDir', dest='pathOutputDir', type=str, required=False)
parser.add_argument('--mode', dest='mode', type=str, required=False)
parser.add_argument('--lMinCl', dest='lMinCl', type=int, required=False)
parser.add_argument('--lMaxCl', dest='lMaxCl', type=int, required=False)
parser.add_argument('--nBinsCl', dest='nBinsCl', type=int, required=False)
parser.add_argument('--binSpacing', dest='binSpacing', type=str, required=False)
parser.add_argument('--binWeighting', dest='binWeighting', type=str, required=False)
parser.add_argument('--nameOutputClFile', dest='nameOutputClFile', type=str, required=False)
parser.add_argument('--spins', dest='spins', type=list, required=False)
parser.add_argument('--pixWindow', dest='pixWindow', type=list, required=False)
parser.add_argument('--pathMap1', dest='pathMap1', type=str, required=False)
parser.add_argument('--pathMask1', dest='pathMask1', type=str, required=False)
parser.add_argument('--pathMap2', dest='pathMap2', type=str, required=False)
parser.add_argument('--pathMask2', dest='pathMask2', type=str, required=False)
parser.add_argument('--pathClTheory', dest='pathClTheory', type=str, required=False)


### END OF PARSER ###



def read_cl(config):

   #!!! Only implemented for scalar fields   
   logger.info('Reading cls from '+config['pathClTheory'])
   data = np.genfromtxt(config['pathClTheory'])
   # check that the lmax requested is in the input file
   if data[-1,0]<config['lMaxCl']:
      raise ValueError('The lmax required is higher than in the cl input file')
   # interpolate the input cl, in case it was not given at each ell, 
   # or did not start at ell=0.
   fClTheory = interp1d(data[:,0], data[:, 1], kind='linear', bounds_error=False, fill_value=0.)
   
   return fClTheory
   




if __name__ == '__main__':

   args = parser.parse_args()

   # read parameters from yaml file
   config = yaml.load(open(args.path2config))
   logger.info('Read config from {}.'.format(args.path2config))
   
   # Overwrite options in yaml file with optional input,
   # if needed
   argDict = args.__dict__
   for key in argDict:
      if config.has_key(key) and argDict[key] is not None:
         config[key] = argDict[key] 


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

      if config['pathMask1'] is None:
         mask1 = np.ones_like(map1)
      else:
         mask1 = hp.read_map(config['pathMask1'])
      logger.info('Read mask1 from {}.'.format(config['pathMask1']))
      
      
      if config['pathMask2'] is None:
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
      

      # choose the maximum multipole in map to compute coupling matrix
      lMaxMap = min(3.*nSide-1, config['lMaxMap'])
      # Generate the ell bins
      #ells = np.arange(config['lMaxCl']+1)
      ells = np.arange(lMaxMap+1)
      weights = np.zeros_like(ells, dtype='float64')
      bpws = -1*np.ones_like(ells)
      # Careful: bins start at zero
      for i in range(config['nBinsCl']):
         bpws[lEdges[i]:lEdges[i+1]] = i
         if config['binWeighting']=='uniform':
            weights[lEdges[i]:lEdges[i+1]] = 1.
         elif config['binWeighting']=='numModes':
            weights[lEdges[i]:lEdges[i+1]] = 2. * ells[lEdges[i]:lEdges[i+1]] + 1.

      b = nmt.NmtBin(nSide, bpws=bpws, ells=ells, weights=weights, lmax=lMaxMap)
      # The effective sampling rate for these bandpowers can be obtained calling:
      ells_decoupled = b.get_effective_ells()

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
      tStart = time()
      wsp = nmt.NmtWorkspace()
      wsp.compute_coupling_matrix(f1, f2, b)
      tStop = time()
      logger.info('took '+str((tStop-tStart)/60.)+' min')

      # Compute pseudo-Cls
      logger.info('Computing (coupled) pseudo-cls')
      cl_coupled = nmt.compute_coupled_cell(f1, f2)
      # Uncoupling pseudo-Cls
      # For one spin-0 field and one spin-2 field, NaMaster gives: n_cls=2, [C_TE,C_TB]
      # For two spin-2 fields, NaMaster gives: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]
      logger.info('Decoupling cls.')
      cl_decoupled = wsp.decouple_cell(cl_coupled)
      # array of measured cl to be saved to file
      cl_out = np.vstack((ells_decoupled, cl_decoupled))
      
      # read the theory cl if requested
      if config['pathClTheory']<>'None':
         # read the theory Cl
         fClTheory = read_cl(config)
         clTheory = [fClTheory(ells)]
         # couple with the mask, bin, then decouple
         clTheory_decoupled = wsp.decouple_cell(wsp.couple_cell(clTheory))
         # array of measured cl to be saved to file
         clTheory_out = np.vstack((ells_decoupled, clTheory_decoupled))
         # save it to file


   # Check that output directory exists, create it if not
   pathOutputDir = config['pathOutputDir']
   if not os.path.isdir(pathOutputDir):
      try:
         os.makedirs(pathOutputDir)
         logger.info('Created directory {}.'.format(pathOutputDir))
      except:
         logger.info('Directory {} already exists.'.format(pathOutputDir))
         pass

   # Save measured cl to file
   path = pathOutputDir + "/" + config['nameOutputClFile'] + "_measuredcl.txt"
   np.savetxt(path, cl_out.T)
   logger.info('Written measured cls to {}.'.format(path))

   # Save the theory cl to file if requested
   if config['pathClTheory']<>'None':
      path = pathOutputDir + "/" + config['nameOutputClFile'] + "_ theorycl.txt"
      np.savetxt(path, clTheory_out.T)
      logger.info('Written theory cls to {}.'.format(path))
       

   # plot to check
   if True:

      # Quick anafast
      clHp = hp.anafast(map1, map2)
      ellHp = np.arange(len(clHp))
      clTheoryHp = fClTheory(ellHp)
      lCenBinnedHp, lEdges, binIndices = stats.binned_statistic(ellHp, ellHp, statistic='mean', bins=20)
      ClBinnedHp, lEdges, binIndices = stats.binned_statistic(ellHp, clHp, statistic='mean', bins=20)
      ClTheoryBinnedHp, lEdges, binIndices = stats.binned_statistic(ellHp, clTheoryHp, statistic='mean', bins=20)
      


      fig=plt.figure(0)
      ax=fig.add_subplot(111)
      #
      ax.plot(ells_decoupled, cl_decoupled[0], 'bx', label=r'Measured, decoupled')
      ax.plot(ells_decoupled, -cl_decoupled[0], 'rx')
      ax.plot(ells_decoupled, clTheory_decoupled[0], '.', label=r'Theory, binned \& decoupled')
      ax.plot(ells, fClTheory(ells), label=r'Theory')
      #
      ax.set_yscale('log', nonposy='clip')
      ax.legend(loc=1)
      ax.set_xlabel(r'$\ell$')
      ax.set_ylabel(r'$C_\ell$')

      fig=plt.figure(1)
      ax=fig.add_subplot(111)
      #

      ax.plot(lCenBinnedHp, ClBinnedHp / ClTheoryBinnedHp - 1., 'g.')

      ax.plot(ells_decoupled, cl_decoupled[0]/clTheory_decoupled[0] -1., 'bx')
      ax.axhline(0., color='k')
      #
      ax.set_xlabel(r'$\ell$')
      #ax.set_ylabel(r'$C_\ell^\text{measured} / C_\ell^\text{th binned} - 1$')



      plt.show()
