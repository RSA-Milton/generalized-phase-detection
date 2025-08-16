import sys
print(f'Python: {sys.version}')

import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')

import keras
print(f'Keras: {keras.__version__}')

import numpy as np
print(f'NumPy: {np.__version__}')

import obspy
print(f'ObsPy: {obspy.__version__}')

# Verificar backend
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
print(f'Backend: {keras.backend.backend()}')
