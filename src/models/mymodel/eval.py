#!/usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import torch

class MyEvaluator:
    def __init__(self, device):
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, model, test_data):
        logger.info('Test start')
        model.eval()
        features = test_data.get_features().view(len(test_data),1 , 28, 28).to(self.device)
        labels = test_data.get_labels().to(self.device)

        predictions = model.predict(features)
        corrects = torch.argmax(predictions, 1) == labels
        accuracy = corrects.float().mean()

        return accuracy.item()