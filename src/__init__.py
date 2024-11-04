"""
BrainNormativeCVAE

A library for running cVAE-based normative modeling on brain imaging data.
This implementation extends the normative modelling framework by 
Lawry Aguila et al. (2022) (https://github.com/alawryaguila/normativecVAE),
with refinements in both the model architecture and inference approach.

References:
    Lawry Aguila, A., Chapman, J., Janahi, M., Altmann, A. (2022). 
    Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases. 
    GitHub Repository, https://github.com/alawryaguila/normativecVAE
"""


from . import models
from . import training
from . import inference
from . import utils

__version__ = '0.1.0'
__author__ = 'M. Ho, Y. Song, P. Sachdev, J. Jiang, W. Wen'
__credits__ = ['A. Lawry Aguila', 'J. Capman', 'M. Janahi', 'A. Altman']