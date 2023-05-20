#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class MyModel(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super(MyModel, self).__init__()

        # initialize variables
        self.in_dim = in_dim
        self.out_dim = out_dim

        # initialize layers
        self.linear = torch.nn.Linear(7*7*64, self.out_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


    def forward(self, features, labels):
        out = self.layer1(features)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        hypothesis = self.linear(out)
        loss = self.criterion(hypothesis, labels)

        return loss

    def predict(self, features):
        out = self.layer1(features)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) 
        scores = self.linear(out)
        return torch.nn.functional.softmax(scores, dim=1)
