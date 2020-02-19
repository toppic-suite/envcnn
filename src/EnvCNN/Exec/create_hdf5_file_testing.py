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
import h5py
import EnvCNN.Data.train_model_util as util
import EnvCNN.Data.hdf5_util as h5_util

data_dir = sys.argv[1]

params_dict = util.get_param_dict(data_dir) ## Read params_dict
labels_dict = util.get_label_dict(data_dir) ## Read labels file
util.shortlist_dictionaries(params_dict, labels_dict) ## removes all annotations other than B and Y. 
data_files, labels = util.shuffle_data_test(labels_dict) 
train_data_files, validation_data_files, test_data_files, train_labels, validation_labels, test_labels = util.split_data_single_2(data_files, labels)

hdf5_path = "dataset_test.hdf5"
hdf5_file = h5py.File(hdf5_path, mode='w') 
h5_util.write_hdf5_categories(300, 6, hdf5_file, train_data_files, validation_data_files, test_data_files)
h5_util.write_labels(hdf5_file, train_labels, validation_labels, test_labels)
h5_util.write_params(hdf5_file, params_dict, train_data_files, validation_data_files, test_data_files)
h5_util.write_train_matrix(hdf5_file, train_data_files, data_dir)
h5_util.write_val_matrix(hdf5_file, validation_data_files, data_dir)
h5_util.write_test_matrix(hdf5_file, test_data_files, data_dir)
hdf5_file.close()
