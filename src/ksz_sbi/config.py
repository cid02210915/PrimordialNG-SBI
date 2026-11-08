import os
import glob
import pickle
import numpy as np
import healpy as hp
import torch
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding

nbins = 8
ell_max = 32
ells = np.arange(ell_max + 1)

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)

print("NPIX =", NPIX)