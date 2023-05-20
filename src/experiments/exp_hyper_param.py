'''
batch_size: batch_size for train 
split: train-test split used for the dataset
'''
batch_size = 100
split = 1



'''
gpu: gpu number to use
device: cuda or cpu
seed: an integer
'''
gpu = 3
seed = -1



'''
model related parameters
dropout: dropout probability for GCN hidden layer
epochs: number of training epochs
checkpoint: whether use checkpoint or not
path: checkpoint's diretory
'''
dropout = 0.5
epochs = 10
checkpoint = False
path = "./experiments/checkpoint/model_5.pt"


'''
parameters for optimisation
rate: learning rate
decay: weight decay
'''
learning_rate = 0.01
decay = 0.0005



'''
model for hypergraph neral network
'''
model="mymodel"




import os
import inspect
import configargparse
from configargparse import YAMLConfigFileParser
import yaml



def parse(cfg):
    """
    adds and parses arguments / hyperparameters
    """
    default = os.path.join(current(), "config", cfg + ".yml")

    p = configargparse.ArgParser(config_file_parser_class = YAMLConfigFileParser, default_config_files=[default])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')
    p.add('--batch_size', type=int, default=batch_size, help='batch_size for train')
    p.add('--split', type=int, default=split, help='train-test split used for the dataset')
    p.add('--dropout', type=float, default=dropout, help='dropout probability for GCN hidden layer')
    p.add('--learning_rate', type=float, default=learning_rate, help='learning rate')
    p.add('--decay', type=float, default=decay, help='weight decay')
    p.add('--epochs', type=int, default=epochs, help='number of epochs to train')
    p.add('--checkpoint', type=bool, default=checkpoint, help='whether use checkpoint or not')
    p.add('--path', type=str, default=path, help='checkpoint dir')
    p.add('--gpu', type=int, default=gpu, help='gpu number to use')
    p.add('--seed', type=int, default=seed, help='seed for randomness')
    p.add('-f') # for jupyter default
    p.add('--model', type=str, default=model, help='model for Hypergraph neural network')
    p.add('--cfg', type=str, default=cfg, help='cfg file name')
    return p.parse_args()

def current():
    """
    return : the current dir path
    """
    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    head, tail = os.path.split(current)
    
    return head