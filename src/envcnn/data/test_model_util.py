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
import keras
import numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import EnvCNN.Data.train_model_util as train_model_util 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def load_model(model_directory):
  """ Load model using json file and its weight using h5 file. Both files must be present in model_directory."""
  exists = os.path.isfile(os.path.join(model_directory,"model.json")) and os.path.isfile(os.path.join(model_directory,"model.h5"))
  if exists:
    json_file = open(os.path.join(model_directory,"model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
  else:
    raise Exception("Error: model files not found. Please train the CNN model.")
  print("Loaded model from disk")
  return loaded_model

def generate_roc_curve(output_directory, predicted_label_probabilities, topfd_score, assigned_label, file_name="ROC.png"):
  auc = roc_auc_score(assigned_label, predicted_label_probabilities)
  topfd_auc = roc_auc_score(assigned_label, topfd_score)
  print('EnvCNN_AUC: %.4f' % auc)
  print('TopFD_AUC: %.4f' % topfd_auc)
  # plot roc curve
  fpr, tpr, thresholds = roc_curve(assigned_label, predicted_label_probabilities)
  topfd_fpr, topfd_tpr, topfd_thresholds = roc_curve(assigned_label, topfd_score)
  plot_roc(fpr, tpr, topfd_fpr, topfd_tpr, output_directory, file_name)

def plot_roc(fpr, tpr, topfd_fpr, topfd_tpr, graph_directory, file_name):
  # plot the roc curve for the model
  plt.figure()
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.plot(topfd_fpr, topfd_tpr, marker='.')
  plt.ylabel('True positive rate')
  plt.xlabel('False positive rate')
  plt.legend(['Reference', 'EnvCNN', 'MS-Deconv'], loc='lower right')
  roc_file_name = os.path.join(graph_directory, file_name) 
  plt.savefig(roc_file_name, dpi=1500)
  plt.close()

def generate_roc_curve_test(output_directory, predicted_label_probabilities, assigned_label, file_name="ROC_Curve_test.png"):
  auc = roc_auc_score(assigned_label, predicted_label_probabilities)
  print('EnvCNN_AUC: %.4f' % auc)
  # plot roc curve
  fpr, tpr, thresholds = roc_curve(assigned_label, predicted_label_probabilities)
  plot_roc_test(fpr, tpr, output_directory, file_name)

def plot_roc_test(fpr, tpr, graph_directory, file_name):
  plt.figure()
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.legend(['Reference', 'EnvCNN'], loc='lower right')
  roc_file_name = os.path.join(graph_directory, file_name) 
  plt.savefig(roc_file_name, dpi=250)
  plt.close()
  
def get_anno_list():
  anno_list = []
  anno_list.append(["B", "B: B-Ions", "B-Ions"])
  anno_list.append(["Y", "Y: Y-Ions", "Y-Ions"])
  return anno_list

def get_anno_dist(model_dir, test_data, test_labels, test_params):
  anno_list = get_anno_list()
  anno_history = []
  for anno in anno_list:
    print("** Processing " + anno[0] + "-ions **")
    x_train, y_train = train_model_util.shortlist_data(test_data, test_labels, test_params, anno[0])
    model = keras.models.load_model(os.path.join(model_dir, "model.h5"))
    model.compile(loss = "binary_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=5e-05))
    history = model.evaluate(x_train, y_train, verbose=0)
    anno_history.append(history)
    predictions = model.predict(x_train, verbose=0)
    plot_prediction_dist(predictions, anno[1], anno[2])
    print("Loss:", history[0], "Accuracy:", history[1])

def get_label_distribution(output_dir, x_train, y_train, file_name="Labels_Distribution.png"):
  condition = y_train == 1
  positive_data = x_train[condition]
  negative_data = x_train[~condition]

  positive_labels = y_train[condition]
  negative_labels = y_train[~condition]
  model = keras.models.load_model(os.path.join(output_dir, "model.h5"))
  positive_history = model.evaluate(positive_data, positive_labels, verbose=0)
  print("** Positive Envelopes **")
  print("Loss:", positive_history[0], "Accuracy:", positive_history[1])
  negative_history = model.evaluate(negative_data, negative_labels, verbose=0)
  print("** Negative Envelopes **")
  print("Loss:", negative_history[0], "Accuracy:", negative_history[1])
  
  ## Label Distribution
  positive_label_probabilities = model.predict(positive_data)
  negative_label_probabilities = model.predict(negative_data)
  plt.figure()
  sns.distplot(positive_label_probabilities, bins=50, kde=False, label="Positive")
  sns.distplot(negative_label_probabilities, bins=50,  kde=False, label="Negative")
  plt.legend(['Positive Labels', 'Negative Labels'], loc='upper right')
  plt.ylabel('# of Envelopes')
  plt.xlabel('Probability Score')
  plt.title("Labels Distribution")
  plt.savefig(file_name, dpi=250)
  plt.close()
  
def plot_prediction_dist(label_probabilities, legend, graph_file_name):
  plt.figure()
  sns.distplot(label_probabilities,  kde=False, label=legend)
  plt.legend([legend], loc='upper right')
  plt.title("Positive Labels Distribution")
  plt.ylabel('# of Envelopes')
  plt.xlabel('Probability Score')
  plt.savefig(graph_file_name , dpi=250)
  plt.close()

def get_charge_stats(output_dir, x_train, y_train):
  model = keras.models.load_model(os.path.join(output_dir, "model.h5"))
  for charge in range(1, 31):
    x_train_charge_list = []
    y_train_charge_list = []
    for idx in range(0,len(y_train)):
      c = x_train[idx,:,2]
      if charge in c:
        x_train_charge_list.append(x_train[idx])
        y_train_charge_list.append(y_train[idx])
    if len(y_train_charge_list) > 1:
      x_train_charge = numpy.array(x_train_charge_list)
      y_train_charge = numpy.array(y_train_charge_list)
      history = model.evaluate(x_train_charge, y_train_charge, verbose=1)
      print("For Charge " + str(charge) + ": " + str(history))
      get_label_distribution(output_dir, x_train_charge, y_train_charge, "Charge_" + str(charge) + "_Labels_Distribution.png")

