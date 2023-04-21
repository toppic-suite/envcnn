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
import os
import torch
import numpy
import math
import h5py
import time
import data.torch_models as models
import data.train_model_util as train_model_util
from pytorchtools import EarlyStopping
from matplotlib import pyplot

# Convert train and validation sets to torch.Tensors and load them to
# DataLoader.
def data_loader(train_x, vali_x, train_y, vali_y, batch_size=512):
  train_x=torch.Tensor(train_x)
  train_y=torch.Tensor(train_y)

  # Create DataLoader for training data
  train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

  vali_x=torch.Tensor(vali_x)
  vali_y=torch.Tensor(vali_y)

  # Create DataLoader for testidation data
  vali_dataset = torch.utils.data.TensorDataset(vali_x, vali_y)
  vali_dataloader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size)

  return train_dataloader, vali_dataloader

def train_epoch(model, loss_fn, train_dataloader):
  total_loss = 0
  train_output=[]
  # Put the model into the training mode
  model.train()
  count = 0
  for batch in train_dataloader:
    count = count + 1
    print("train batch", count, " out of ", len(train_dataloader), end = "\r") 
    # Load batch to the device 
    b_input_ids, b_labels = tuple(t.to(device) for t in batch)
    # Zero out any previously calculated gradients
    optimizer.zero_grad()

    # Perform a forward pass. This will return logits
    results = model(b_input_ids)
    if torch.cuda.is_available():
      results_numpy=results.detach().cpu().numpy()
    else:
      results_numpy=results.detach().numpy()
    train_output.append(results_numpy)
       
    # Compute loss and accumulate the loss values
    loss = loss_fn(results, b_labels)
    #print("loss", loss.item())
    total_loss += loss.item()

    # Perform a backward pass to calculate gradients
    loss.backward()

    # Update parameters
    optimizer.step()
    
  # Calculate the average loss over the entire training data
  avg_train_loss = total_loss / len(train_dataloader)
  
  return avg_train_loss, train_output

def comp_accuracy(results, b_labels):
  #print(results)
  accuracy = 0
  for i in range(results.shape[0]):
    pred = 0
    if (results[i][1] > results[i][0]):
      pred = 1
    if pred == b_labels[i]:
      accuracy = accuracy + 1
  #print(b_labels)
  #print(accuracy)
  return accuracy/results.shape[0]

def validate_epoch(model, loss_fn, vali_dataloader):
  # Put the model into the evaluation mode. The dropout layers are disabled
  # during the test time.
  model.eval()

  # Tracking variables
  total_loss = 0 
  total_accuracy = 0
  vali_output= []

  count = 0
  for batch in vali_dataloader:
    count = count + 1
    print("validation batch", count, " out of ", len(vali_dataloader), end = "\r") 
    b_input_ids, b_labels = tuple(t.to(device) for t in batch)

    # Perform a forward pass. This will return logits
    with torch.no_grad():
      results = model(b_input_ids)
    if torch.cuda.is_available():
      results_numpy=results.detach().cpu().numpy()
    else:
      results_numpy=results.detach().numpy()
    vali_output.append(results_numpy)
       
    # Compute loss and accumulate the loss values
    loss = loss_fn(results, b_labels)
    total_loss = total_loss + loss.item()
    
    accuracy = comp_accuracy(results, b_labels)
    total_accuracy = total_accuracy + accuracy

  # Compute the average accuracy and loss over the validation set.
  avg_vali_loss = total_loss / len(vali_dataloader) 
  avg_vali_accuracy = total_accuracy / len(vali_dataloader) 
  return avg_vali_loss, avg_vali_accuracy, vali_output

def plot_losses(logs, output):
  pyplot.subplot()  
  pyplot.plot(logs['train_loss'])   
  pyplot.plot(logs['validation_loss'])
  pyplot.title('model loss')  
  pyplot.ylabel('loss')  
  pyplot.xlabel('epoch')  
  #pyplot.ylim([0,0.03])
  pyplot.legend(['train','validation'], loc='upper right')
  pyplot.savefig(output + '_loss.png')
  pyplot.close()
  


# Train the model
def train(model, optimizer, loss_fn, x_train, x_test, y_train, y_test, batch_size, epochs):
  # Tracking best validation accuracy
  logs={}
  logs['train_loss']=[]
  logs['validation_loss']=[]
  # Set best loss to a large number
  best_loss = 100000

  train_dataloader, vali_dataloader = data_loader(x_train, x_test, y_train, y_test, batch_size)
  early_stopping = EarlyStopping(patience=30, verbose=True, path = "early_stop_checkpoint.pt")

  # Start training loop
  print("Start training...\n")
  print(f"{'Epoch':^7} | {'Train Loss':^16} |  {'Validation Loss':^16} | {'Valid Accuracy':^16} | {'Elapsed':^9}")
  print("-"*60)
  for epoch_i in range(epochs):
    # Tracking time 
    t0_epoch = time.time()
    # Training
    if train_dataloader is not None:
      avg_train_loss, train_output= train_epoch(model, loss_fn, train_dataloader)
      logs['train_loss'].append(avg_train_loss)
    # Evaluation
    if vali_dataloader is not None:
      avg_vali_loss, avg_vali_accuracy, vali_output= validate_epoch(model, loss_fn, vali_dataloader)
      logs['validation_loss'].append(avg_vali_loss)
      # Track the best accuracy
      if avg_vali_loss < best_loss:
        best_loss = avg_vali_loss

    # Print performance over the entire training data
    time_elapsed = time.time() - t0_epoch
    print(f"{epoch_i + 1:^7} | {avg_train_loss:^16.6f} |  {avg_vali_loss:^16.6f} | {avg_vali_accuracy:^16.6f} | {time_elapsed:^9.2f}")

    early_stopping(avg_vali_loss, model)
        
    if early_stopping.early_stop:
      print("Early stopping")
      break

  print("\n")
  print(f"Training complete! Best test loss: {best_loss:.4f}.")

  plot_losses(logs, "result")

        
# Main
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

t0 = time.time()
data_dir = sys.argv[1]
output_dir = train_model_util.create_output_directory("output")

hdf5_file = h5py.File(data_dir, "r")
num_train_samples = hdf5_file["train_data"].shape[0]
num_val_samples = hdf5_file["val_data"].shape[0]
class_weights = train_model_util.get_class_weight(hdf5_file["train_labels"])

print("train shape: ", num_train_samples)
print("validation shape: ", num_val_samples)
print("class weight:", class_weights)

model=models.EnvCnn() 
print(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-05)
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 128
epochs = 200

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

train(model, optimizer, loss_fn, torch_x_train, torch_x_vali, torch_y_train,
        torch_y_vali, batch_size, epochs)

t1 = time.time()
total = t1-t0
print("Running time:", total)
