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

#!/usr/bin/env python3

import sys
import os
import keras
import math
import h5py
import EnvCNN.Data.models as models
import EnvCNN.Data.train_model_util as train_model_util
import time

t0 = time.time()
data_dir = sys.argv[1]
batch_size = 128
output_dir = train_model_util.create_output_directory("output")

hdf5_file = h5py.File(data_dir, "r")
num_train_samples = hdf5_file["train_data"].shape[0]
num_val_samples = hdf5_file["val_data"].shape[0]
class_weights = train_model_util.get_class_weight(hdf5_file["train_labels"])

print("train_shape: ", num_train_samples)
print("val_shape: ", num_val_samples)
print(class_weights)


train_gen = train_model_util.Hdf5_generator(hdf5_file["train_data"], hdf5_file["train_labels"], batch_size, num_train_samples)
val_gen = train_model_util.Hdf5_generator(hdf5_file["val_data"], hdf5_file["val_labels"], batch_size, num_val_samples)

model = models.vgg()
model.summary()
model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=1e-05))
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "model.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min') #Save Model Checkpoint
history = model.fit_generator(train_gen, steps_per_epoch=math.ceil(num_train_samples / batch_size), validation_data=val_gen, validation_steps=math.ceil(num_val_samples / batch_size), epochs=200, verbose=2, class_weight=class_weights, callbacks=[checkpoint, early_stopping])
hdf5_file.close()

train_model_util.save_model(model, output_dir)
train_model_util.print_training_history(history, output_dir)
train_model_util.plot_training_graphs(history, output_dir)

t1 = time.time()
total = t1-t0
print(total)
