#!/usr/bin/env python3

import sys
import h5py

data_file_name = sys.argv[1]
output_file_name =sys.argv[2]

hdf5_file = h5py.File(data_file_name, "r")
vali_params = hdf5_file["val_params"]

output_file = open(output_file_name, "w")
for i in range(len(vali_params)):
  para = vali_params[i].decode("utf-8")
  output_file.write(para)
  output_file.write("\n")

output_file.close()
