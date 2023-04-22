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

import os
import h5py
import numpy as np

def write_hdf5_categories(d1, d2, hdf5_file, train_addrs, val_addrs, test_addrs):
  train_shape = (len(train_addrs), d1, d2)
  val_shape = (len(val_addrs), d1, d2)
  test_shape = (len(test_addrs), d1, d2)
  
  hdf5_file.create_dataset("train_data", train_shape, np.float)
  hdf5_file.create_dataset("val_data", val_shape, np.float)
  hdf5_file.create_dataset("test_data", test_shape, np.float)
  
  hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
  hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
  hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
  
  dt = h5py.special_dtype(vlen=str)
  hdf5_file.create_dataset("train_params", (len(train_addrs),), dt)
  hdf5_file.create_dataset("val_params", (len(val_addrs),), dt)
  hdf5_file.create_dataset("test_params", (len(test_addrs),), dt)

def write_labels(hdf5_file, train_labels, val_labels, test_labels):
  hdf5_file["train_labels"][...] = train_labels
  hdf5_file["val_labels"][...] = val_labels
  hdf5_file["test_labels"][...] = test_labels

def write_params(hdf5_file, params, train_data_files, validation_data_files, test_data_files):
  hdf5_file["train_params"][...] = [x + ',' + str(params[x][0]) + ',' + str(params[x][1]) + ',' + str(params[x][2]) for x in train_data_files]
  hdf5_file["val_params"][...] = [x + ',' + str(params[x][0]) + ',' + str(params[x][1]) + ',' + str(params[x][2]) for x in validation_data_files]
  hdf5_file["test_params"][...] = [x + ',' + str(params[x][0]) + ',' + str(params[x][1]) + ',' + str(params[x][2]) for x in test_data_files]

def write_data_files(hdf5_file, train_data_files, validation_data_files, test_data_files):
  hdf5_file["train_data_files"][...] = train_data_files
  hdf5_file["val_data_files"][...] = validation_data_files
  hdf5_file["test_data_files"][...] = test_data_files  
  
def write_train_matrix(hdf5_file, train_addrs, data_dir):
  # loop over train addresses
  for i in range(len(train_addrs)):
    # print how many matrices are saved every 1000 matrices
    if i % 1000 == 0 and i > 1:
      print("Train data: {}/{}".format(i, len(train_addrs)))
    # read a matrix
    file_name = train_addrs[i]
    train_data = np.genfromtxt(data_dir + os.sep + file_name, delimiter=',')
    # save the matrix
    hdf5_file["train_data"][i, ...] = train_data[None]

def write_val_matrix(hdf5_file, val_addrs, data_dir):
  # loop over validation addresses
  for i in range(len(val_addrs)):
    # print how many matrices are saved every 1000 matrices
    if i % 1000 == 0 and i > 1:
      print("Validation data: {}/{}".format(i, len(val_addrs)))
    # read a matrix
    file_name = val_addrs[i]
    val_data = np.genfromtxt(data_dir + os.sep + file_name, delimiter=',')
    # save the matrix
    hdf5_file["val_data"][i, ...] = val_data[None]

def write_test_matrix(hdf5_file, test_addrs, data_dir):
  # loop over test addresses
  for i in range(len(test_addrs)):
    # print how many matrices are saved every 1000 matrices
    if i % 1000 == 0 and i > 1:
      print("Test data: {}/{}".format(i, len(test_addrs)))
    # read a matrix
    file_name = test_addrs[i]
    test_data = np.genfromtxt(data_dir + os.sep + file_name, delimiter=',')
    # save the matrix
    hdf5_file["test_data"][i, ...] = test_data[None]
