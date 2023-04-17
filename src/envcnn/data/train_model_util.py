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
import random
import numpy
import pandas
import sklearn.model_selection
from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from random import shuffle
import seaborn as sns

def Hdf5_generator(X, y, batch_size, nsamples):
  start_idx = 0
  while True:
    if start_idx + batch_size > nsamples:
      start_idx = 0
    x_batch = X[start_idx:start_idx+batch_size, ...][:,:,0:5]
    y_batch = y[start_idx:start_idx+batch_size, ...]
    start_idx += batch_size
    yield numpy.asarray(x_batch), numpy.asarray(y_batch)

def get_class_weight(y_train):
  label_weights = class_weight.compute_class_weight(class_weight='balanced', classes=numpy.unique(y_train), y=y_train)
  class_weight_dict = dict(enumerate(label_weights))
  return class_weight_dict

def get_lables_train_valid_filenames(data_dir):
  label_file = os.path.join(data_dir, "label.csv")
  file_data = pandas.read_csv(label_file, delimiter=',', header=None)
  data_files = file_data[0].tolist()
  labels = file_data[1].tolist()
  train_data_files, validation_data_files, train_labels, validation_labels = sklearn.model_selection.train_test_split(data_files, labels, test_size=0.2, shuffle=False)
  return train_data_files, validation_data_files, train_labels, validation_labels

def split_data_single(data_files, labels):
  train_data_files = data_files[0:int(0.8*len(data_files))]
  train_labels = labels[0:int(0.8*len(labels))]
  validation_data_files = data_files[int(0.8*len(data_files)):]
  validation_labels = labels[int(0.8*len(labels)):]
  test_data_files = []
  test_labels = []
  return train_data_files, validation_data_files, test_data_files, train_labels, validation_labels, test_labels
  
def split_data_single_2(data_files, labels):
  train_data_files = []
  train_labels = []
  validation_data_files = []
  validation_labels = []
  test_data_files = data_files
  test_labels = labels
  return train_data_files, validation_data_files, test_data_files, train_labels, validation_labels, test_labels

def split_data(data_files, labels):
  train_data_files = data_files[0:int(0.6*len(data_files))]
  train_labels = labels[0:int(0.6*len(labels))]
  validation_data_files = data_files[int(0.6*len(data_files)):int(0.8*len(data_files))]
  validation_labels = labels[int(0.6*len(labels)):int(0.8*len(labels))]
  test_data_files = data_files[int(0.8*len(data_files)):]
  test_labels = labels[int(0.8*len(labels)):]
  return train_data_files, validation_data_files, test_data_files, train_labels, validation_labels, test_labels

def split_data_train(data_files, labels):
  train_data_files = data_files[0:int(0.8*len(data_files))]
  train_labels = labels[0:int(0.8*len(labels))]
  validation_data_files = data_files[int(0.8*len(data_files)):]
  validation_labels = labels[int(0.8*len(labels)):]
  return train_data_files, validation_data_files, train_labels, validation_labels
  
def shuffle_data(labels_dict):
  data_files = list(labels_dict.keys())
  labels = list(labels_dict.values())
  c = list(zip(data_files, labels))
  shuffle(c)
  data_files, labels = zip(*c)
  return list(data_files), list(labels)

def shuffle_data_test(labels_dict):
  data_files = list(labels_dict.keys())
  labels = list(labels_dict.values())
  c = list(zip(data_files, labels))
  #shuffle(c)
  data_files, labels = zip(*c)
  return list(data_files), list(labels)

def get_param_dict(data_dir):
  param_file = os.path.join(data_dir, "parameters.csv")
  params_pd = pandas.read_csv(param_file, delimiter=',', header=None)
  params = dict([(i,[a,b,c]) for i, a,b,c in zip(params_pd[0], params_pd[1],params_pd[2],params_pd[3])])
  return params

def get_label_dict(data_dir):
  label_file = os.path.join(data_dir, "label.csv")
  label_pd = pandas.read_csv(label_file, delimiter=',', header=None)
  labels = dict([(i,a) for i, a in zip(label_pd[0], label_pd[1])])
  return labels

def shortlist_dictionaries(params, labels):
  rm_keys = []
  for k, v in params.items():
    if v[1] == "C" or v[1] == "Z+1" or v[1] == "A" or v[1] == "X" or v[1] == "B-Water" or v[1] == "Y-Water" or v[1] == "B-Ammonia" or v[1] == "Y-Ammonia" or v[1] == "B-1" or v[1] == "Y-1" or v[1] == "B+1" or v[1] == "Y+1":
      rm_keys.append(k)
  for k in rm_keys: 
    del params[k]
    del labels[k]

def shuffle_split_data(a, b):
  combined = list(zip(a, b))
  random.shuffle(combined)
  a[:], b[:] = zip(*combined)

def update_dict(data_dir, params):
  new_keys = []
  for k, v in params.items():
    new_keys.append(os.path.join(data_dir, k[2:]))
  d1 = dict( zip( list(params.keys()), new_keys) )
  return {d1[oldK]: value for oldK, value in params.items()}

def create_output_directory(dir_name):
  ouput_dir = os.path.join(os.getcwd(), dir_name)
  if os.path.isdir(ouput_dir) == False:
    os.mkdir(ouput_dir)
  return ouput_dir

def shortlist_data(test_data, test_labels, test_params, anno):
  shortlisted_test_data = []
  shortlisted_test_labels = []
  for idx in range(0, len(test_labels)):
    param = test_params[idx].split(',')
    if param[2] == anno:
      shortlisted_test_data.append(test_data[idx])
      shortlisted_test_labels.append(test_labels[idx])
  x_train = numpy.stack(shortlisted_test_data)
  y_train = numpy.array(shortlisted_test_labels)
  return x_train, y_train

def print_training_history(history, ouput_dir):
  """ Print Losses and Accuracy from model training and validation into CSV file """
  training_file_name = os.path.join(ouput_dir, "TrainingHistory.txt")
  f = open(training_file_name, "w")
  f.writelines("Training Data History") # Write a string to a file
  f.writelines("\nTraining_Accuracy :" + str(history.history['acc']) + " \nValidation_Accuracy :" + str(history.history['val_acc']) ) # Write a string to a file
  f.writelines("\nTraining_Loss :" + str(history.history['loss']) + " \nValidation_Loss :" + str(history.history['val_loss']) ) # Write a string to a file
  f.close()

def plot_training_graphs(history, ouput_dir):
  """ Calls functions to draw loss and accuracy graphs """
  _plot_loss_graph(history, ouput_dir)
  _plot_accuracy_graph(history, ouput_dir)

def _plot_loss_graph(history, ouput_dir):
  """ Plot Losses from model training and validation """
  graph_file_name = os.path.join(ouput_dir, "ModelLoss.png")
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')  
  plt.ylabel('loss')  
  plt.xlabel('epoch')  
  plt.legend(['train', 'test'], loc='upper left')  
  plt.savefig(graph_file_name, dpi=250)
  plt.close()

def _plot_accuracy_graph(history, ouput_dir):
  """ Plot Accuracy from model training and validation """
  graph_file_name = os.path.join(ouput_dir, "ModelAccuracy.png")   
  plt.figure()
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(graph_file_name, dpi=250)
  plt.close()
