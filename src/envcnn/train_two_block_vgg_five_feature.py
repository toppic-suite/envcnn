#!/usr/bin/env python3

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

import sys
import time
import numpy
import h5py
import torch

import util.train_model_util as train_model_util
import util.train_torch_model_hdf5 as train_model

import torch_model.torch_two_block_vgg_five_feature as test_model

# Main
t0 = time.time()
data_dir = sys.argv[1]

hdf5_file = h5py.File(data_dir, "r")
num_train_samples = hdf5_file["train_data"].shape[0]
num_val_samples = hdf5_file["val_data"].shape[0]
weights_dict = train_model_util.get_class_weight(hdf5_file["train_labels"])
weights = list(weights_dict.values())

print("train shape: ", num_train_samples)
print("validation shape: ", num_val_samples)
print("class weight:", weights)

if torch.cuda.is_available():       
  device = torch.device("cuda")
  print(f'There are {torch.cuda.device_count()} GPU(s) available.')
  print('Device name:', torch.cuda.get_device_name(0))
  class_weights = torch.FloatTensor(weights).cuda()

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")
  class_weights = torch.FloatTensor(weights)

model=test_model.EnvCnn() 
print(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-05)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
batch_size = 128
epochs = 100

X = hdf5_file["train_data"]
x_train = X[:,:,0:5]
print("x_train shape", x_train.shape)
x_train = x_train.astype(numpy.float32)
torch_x_train = torch.from_numpy(x_train)
# transpose the 2nd and 3rd dimension
torch_x_train = torch_x_train.transpose(1,2)

y_train = hdf5_file["train_labels"][:]
y_train = y_train.astype(numpy.int64)
torch_y_train = torch.from_numpy(y_train)

X = hdf5_file["val_data"]
x_vali = X[:,:,0:5]
print("x_vali shape", x_vali.shape)
x_vali = x_vali.astype(numpy.float32)
torch_x_vali = torch.from_numpy(x_vali)
torch_x_vali = torch_x_vali.transpose(1,2)

y_vali = hdf5_file["val_labels"][:]
y_vali = y_vali.astype(numpy.int64)
torch_y_vali = torch.from_numpy(y_vali)

output_model_file = "two_block_vgg_five_feature.model"

train_model.train(device, model, optimizer, loss_fn, torch_x_train, torch_x_vali, torch_y_train,
                  torch_y_vali, batch_size, epochs, output_model_file)

t1 = time.time()
total = t1-t0
print("Running time:", total)
