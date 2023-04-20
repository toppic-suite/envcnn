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

# Convert train and test sets to torch.Tensors and load them to
# DataLoader.
def data_loader(train_x, test_x, train_y, test_y, batch_size=512):
  train_x=torch.Tensor(train_x)
  train_y=torch.Tensor(train_y)

  # Create DataLoader for training data
  train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

  test_x=torch.Tensor(test_x)
  test_y=torch.Tensor(test_y)

  # Create DataLoader for testidation data
  test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

  return train_dataloader, test_dataloader

# Train the CNN model.
def train(model, optimizer, loss_fn, x_train, x_test, y_train, y_test, batch_size, epochs):
  # Tracking best validation accuracy
  logs={}
  logs['train_loss']=[]
  logs['test_loss']=[]
  # Set best loss to a large number
  best_loss = 100000

  train_dataloader, test_dataloader = data_loader(x_train, x_test, y_train, y_test, batch_size)
  early_stopping = EarlyStopping(patience=30, verbose=True, path = "early_stop_checkpoint.pt")

  # Start training loop
  print("Start training...\n")
  print(f"{'Epoch':^7} | {'Train Loss':^12} |  {'Test Loss':^10} | {'Elapsed':^9}")
  print("-"*60)
  for epoch_i in range(epochs):
    # Tracking time and loss
    t0_epoch = time.time()
    ##tracking the loss for training and validation
    total_loss = 0
    # =======================================
    #               Training
    # =======================================
    train_output=[]
    # Put the model into the training mode
    model.train()

    for step, batch in enumerate(train_dataloader):
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
    logs['train_loss'].append(avg_train_loss)

    # Print performance over the entire training data
    time_elapsed = time.time() - t0_epoch
    test_loss = 0
    print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} |  {test_loss:^10.6f} | {time_elapsed:^9.2f}")
        
        
    """  
        # =======================================
        #               Evaluation
        # =======================================
        if test_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our testidation set.
            test_loss, test_output= evaluate(model, loss_fn,test_dataloader)
            logs['test_loss'].append(test_loss)
            # Track the best accuracy
            if test_loss < best_loss:
                best_loss = test_loss

            
            early_stopping(test_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
    PlotLosses(logs,output,fold)
    
    print("\n")
    print(f"Training complete! Best test loss: {best_loss:.4f}.")
    """


if torch.cuda.is_available():       
    device = torch.device("cpu")
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
x_train = X[:256,:,0:5]
print("x_train shape", x_train.shape)
x_train = x_train.astype(numpy.float32)
torch_x_train = torch.from_numpy(x_train)
# transpose the 2nd and 3rd dimension
torch_x_train = torch_x_train.transpose(1,2)

y_train = hdf5_file["train_labels"][:256]
y_train = y_train.astype(numpy.int64)
torch_y_train = torch.from_numpy(y_train)

X = hdf5_file["val_data"]
x_vali = X[:256,:,0:5]
print("x_vali shape", x_vali.shape)
x_vali = x_vali.astype(numpy.float32)
torch_x_vali = torch.from_numpy(x_vali)
torch_x_vali = torch_x_vali.transpose(1,2)

y_vali = hdf5_file["val_labels"][:256]
y_vali = y_vali.astype(numpy.int64)
torch_y_vali = torch.from_numpy(y_vali)

train(model, optimizer, loss_fn, torch_x_train, torch_x_vali, torch_y_train,
        torch_y_vali, batch_size, epochs)

t1 = time.time()
total = t1-t0
print("Running time:", total)


"""
train_gen = train_model_util.Hdf5_generator(hdf5_file["train_data"], hdf5_file["train_labels"], batch_size, num_train_samples)
val_gen = train_model_util.Hdf5_generator(hdf5_file["val_data"], hdf5_file["val_labels"], batch_size, num_val_samples)

model = models.vgg()
model.summary()
model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=1e-05))
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "model.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min') #Save Model Checkpoint
#history = model.fit_generator(train_gen, steps_per_epoch=math.ceil(num_train_samples / batch_size), validation_data=val_gen, validation_steps=math.ceil(num_val_samples / batch_size), epochs=200, verbose=2, class_weight=class_weights, callbacks=[checkpoint, early_stopping])

history = model.fit(train_gen, steps_per_epoch=math.ceil(num_train_samples / batch_size), validation_data=val_gen, validation_steps=math.ceil(num_val_samples / batch_size), epochs=200, verbose=2, class_weight=class_weights, callbacks=[checkpoint, early_stopping])
hdf5_file.close()

# train_model_util.save_model(model, output_dir)
train_model_util.print_training_history(history, output_dir)
train_model_util.plot_training_graphs(history, output_dir)
"""

