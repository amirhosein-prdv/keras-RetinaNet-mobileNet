
# from tensorflow import keras
# import numpy as np
# from keras_retinanet.preprocessing.csv_generator import CSVGenerator
# from keras_retinanet.utils.eval import evaluate
# from keras_retinanet.utils.gpu import setup_gpu

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras_retinanet.models.mobilenet import CUSTOM_OBJECTS
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# setup_gpu('0')

model_path = '/content/gdrive/MyDrive/retinanet.h5'

# This line must be executed before loading Keras model.
K.set_learning_phase(0)
model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

model.summary()


# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

# inspect the layers operations inside your frozen graph definition and see the name of its input and output tensors
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
# serialize the frozen graph and its text representation to disk.
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pb",
                  as_text=False)

#Optional
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="simple_frozen_graph.pbtxt",
                as_text=True)