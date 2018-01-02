from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K

def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    # X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model

# Steps Involved : 

# Step 1 : Create the model by calling the function above.
# Step 2 : Compile the model by calling. 
model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])
# Step 3 : Train the model on train data by calling. 
model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)
# Step 4 : Test the model on test data by calling. 
model.evaluate(x = ..., y = ...)

# Add layers in Keras :

keras.layers.Add()

import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  
# equivalent to added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

# Similarly support for Multiply, Subtract, Concatnate, Dot etc also exists.

# Average Pooling 2D in Keras :

keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# Average pooling operation for spatial data. 
# pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). (2, 2) will halve the input in both spatial dimension. 
# If only one integer is specified, the same window length will be used for both dimensions.

keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
# Zero-padding layer for 2D input (e.g. picture).
# This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.