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
import h5py
import numpy as np

## Create File
hdf5_path = "dataset.hdf5"
if not os.path.exists(hdf5_path):
  hf = h5py.File(hdf5_path, mode='a') 
  hf.create_dataset("train_data", (0,300, 6), maxshape=(None, 300, 6), dtype=np.float, chunks=True)
  hf.create_dataset("val_data", (0,300, 6), maxshape=(None, 300, 6), dtype=np.float, chunks=True)
  hf.create_dataset("test_data", (0,300, 6), maxshape=(None, 300, 6), dtype=np.float, chunks=True)
  hf.create_dataset("train_labels", (0,), maxshape=(None,), dtype=np.float, chunks=True)
  hf.create_dataset("val_labels", (0,), maxshape=(None,), dtype=np.float, chunks=True)
  hf.create_dataset("test_labels", (0,), maxshape=(None,), dtype=np.float, chunks=True)
  
  dt = h5py.special_dtype(vlen=str)
  hf.create_dataset("train_params", (0,), maxshape=(None,), dtype=dt, chunks=True)
  hf.create_dataset("val_params", (0,), maxshape=(None,), dtype=dt, chunks=True)
  hf.create_dataset("test_params", (0,), maxshape=(None,), dtype=dt, chunks=True)
  hf.close()

## Merge Data
data_dir = sys.argv[1]
source_file = h5py.File(data_dir, "r")
hf = h5py.File(hdf5_path, mode='a') 
## Copy Data from source
if source_file["train_data"].shape[0] > 0:
  hf["train_data"].resize((hf["train_data"].shape[0] + source_file["train_data"].shape[0]), axis = 0)
  hf["train_data"][-source_file["train_data"].shape[0]:] = source_file["train_data"]
  hf["train_labels"].resize((hf["train_labels"].shape[0] + source_file["train_labels"].shape[0]), axis = 0)
  hf["train_labels"][-source_file["train_labels"].shape[0]:] = source_file["train_labels"]
  hf["train_params"].resize((hf["train_params"].shape[0] + source_file["train_params"].shape[0]), axis = 0)
  hf["train_params"][-source_file["train_params"].shape[0]:] = source_file["train_params"]

if source_file["val_data"].shape[0] > 0:
  hf["val_data"].resize((hf["val_data"].shape[0] + source_file["val_data"].shape[0]), axis = 0)
  hf["val_data"][-source_file["val_data"].shape[0]:] = source_file["val_data"]
  hf["val_labels"].resize((hf["val_labels"].shape[0] + source_file["val_labels"].shape[0]), axis = 0)
  hf["val_labels"][-source_file["val_labels"].shape[0]:] = source_file["val_labels"]
  hf["val_params"].resize((hf["val_params"].shape[0] + source_file["val_params"].shape[0]), axis = 0)
  hf["val_params"][-source_file["val_params"].shape[0]:] = source_file["val_params"]

if source_file["test_data"].shape[0] > 0:
  hf["test_data"].resize((hf["test_data"].shape[0] + source_file["test_data"].shape[0]), axis = 0)
  hf["test_data"][-source_file["test_data"].shape[0]:] = source_file["test_data"]
  hf["test_labels"].resize((hf["test_labels"].shape[0] + source_file["test_labels"].shape[0]), axis = 0)
  hf["test_labels"][-source_file["test_labels"].shape[0]:] = source_file["test_labels"]
  hf["test_params"].resize((hf["test_params"].shape[0] + source_file["test_params"].shape[0]), axis = 0)
  hf["test_params"][-source_file["test_params"].shape[0]:] = source_file["test_params"]

hf.close()
