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

import torch.nn as nn

class EnvCnn(nn.Module):
    def __init__(self):  
        super(EnvCnn, self).__init__()
        input_dim = 5
        self.conv1=nn.Conv1d(in_channels = input_dim, #input height
                out_channels = 64, #n_filter
                kernel_size = 3, #filter size
                stride=1,  #filter step
                padding=1)

        self.conv2=nn.Conv1d(in_channels = 64, #input height
                out_channels = 64, #n_filter
                kernel_size = 3, #filter size
                stride=1,  #filter step
                padding=1)

        self.conv3=nn.Conv1d(in_channels = 64, #input height
                out_channels = 128, #n_filter
                kernel_size = 3, #filter size
                stride=1,  #filter step
                padding=1)

        self.conv4=nn.Conv1d(in_channels = 128, #input height
                out_channels = 128, #n_filter
                kernel_size = 3, #filter size
                stride=1,  #filter step
                padding=1)

        self.relu = nn.ReLU()
        self.pool= nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

        # fully connected layer
        self.fc= nn.Sequential(
                nn.Linear(300 * 128, 2048),
                nn.ReLU(),
                nn.Linear(2048,1024),
                nn.ReLU(),
                nn.Linear(1024,2),
                )
        return


    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv3(x)
        x=self.relu(x)
        x=self.conv4(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.flatten(x)

        for layer in self.fc:
            x=layer(x)

        return x

