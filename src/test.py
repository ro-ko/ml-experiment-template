import yaml
import sys
import fire
import os
import inspect
import configargparse
from configargparse import YAMLConfigFileParser
from experiments.exp_hyper_param import current

data = "test"
def currentd():
    """
    return : the current dir path
    """
    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    head, tail = os.path.split(current)
    print(head, tail)
    return head

def main(file="test.yaml"):
    default = os.path.join(current(),data+'.yaml')
    print(default)
    currentd()

if __name__ == "__main__":
    sys.exit(fire.Fire(main))
