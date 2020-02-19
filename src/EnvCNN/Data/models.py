##Copyright (c) 2014 - 2020, The Trustees of Indiana University.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.

#!/usr/bin/python3

import keras
from functools import partial

conv3 = partial(keras.layers.Conv1D, kernel_size=3, strides=1, padding='same', activation='relu')

def _block(in_layer, filters, n_convs):
  vgg_block = in_layer
  for _ in range(n_convs):
    vgg_block = conv3(filters=filters)(vgg_block)
  return vgg_block

def vgg(in_shape=(300, 6)):
  in_layer = keras.layers.Input(in_shape)
  block1 = _block(in_layer, 64, 2)
  pool1 = keras.layers.MaxPool1D()(block1)
  block2 = _block(pool1, 128, 2)
  pool2 = keras.layers.MaxPool1D()(block2)
  block3 = _block(pool2, 256, 2)
  pool3 = keras.layers.MaxPool1D()(block3)
  block4 = _block(pool3, 512, 2)
  pool4 = keras.layers.MaxPool1D()(block4)
  block5 = _block(pool4, 512, 2)
  pool5 = keras.layers.MaxPool1D()(block5)
  flattened = keras.layers.GlobalAvgPool1D()(pool5)
  dense1 = keras.layers.Dense(2048, activation='relu')(flattened)
  dense2 = keras.layers.Dense(1024, activation='relu')(dense1)
  preds = keras.layers.Dense(1, activation='sigmoid')(dense2)
  model = keras.models.Model(in_layer, preds)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
  