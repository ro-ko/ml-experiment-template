#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models.mymodel.model import MyModel
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from utils import createDir


class MyTrainer:
    def __init__(self, device, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    def train_with_hyper_param(self, train_data, hyper_param):

        batch_size = hyper_param['batch_size']
        epochs = hyper_param['epochs']
        learning_rate = hyper_param['learning_rate']
        checkpoint = hyper_param['checkpoint']
        path = hyper_param['path']
        cfg_name = hyper_param['cfg_name']
        checkpoint_dir = './experiments/checkpoint/'+str(cfg_name)
        createDir(checkpoint_dir)

        data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

        total_batches = len(data_loader)
        
        model = MyModel(self.in_dim, self.out_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if checkpoint:
            chkpoint = torch.load(path)
            model.load_state_dict(chkpoint['model_state_dict'])
            optimizer.load_state_dict(chkpoint['optimizer_state_dict'])
            start = chkpoint['epoch']+1
            loss = chkpoint['loss']
        else: start = 1

        model.train()
        writer = SummaryWriter()
        logger.info("Train start")
        pbar = tqdm(range(start, epochs+1), position=0, leave=False, colour='green', desc='epoch')
        for epoch in pbar:
            avg_loss = 0
            for features, labels in tqdm(data_loader, position=1, leave=False, colour='red', desc='batch'):
                # send data to a running device (GPU or CPU)
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                loss = model(features, labels)

                loss.backward()
                optimizer.step()

                avg_loss += loss / total_batches
                
            pbar_str = "epoch: [loss = %.4f]" % loss.item()
            pbar.set_description(pbar_str)
            pbar.write('Epoch {:02}: {:.4} training loss'.format(epoch, loss.item()))
            
            writer.add_scalar("Loss/train", loss, epoch)
            with open(os.path.join("./experiments/log", cfg_name + ".log"), "a") as f:
                f.write(datetime.today().strftime("%Y-%m-%d %H:%M:%S") + ' | Train loss | Epoch {:02}: {:.4} training loss\n'.format(epoch, loss.item()))
            if epoch % 5 == 0:
                torch.save({
                    'epoch' : epoch,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : avg_loss,
                }, checkpoint_dir + '/model_' + str(epoch) + '.pt')
        pbar.close()
        writer.close()

        return model
