
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
from keras.models import load_model
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.models.mobilenet import CUSTOM_OBJECTS
setup_gpu('0')

model_path = '/content/gdrive/MyDrive/retinanet.h5'
# This line must be executed before loading Keras model.
from keras import backend as K

K.set_learning_phase(0)
model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

model.summary()