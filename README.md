# ml-project-template

## Overview
In order to have an efficient research and development environment, it is necessary to have a systematic code structure using an IDE.

- When conducting an experiment on large-scale data, it runs in the background on the server. When the server connection is turned off, the entire experiment is stopped (using shell and some linux tools, you can do it stably)
- You can run multiple experiments in parallel for multiple hyperparameters.(terminal input)
- When launch code through github in the future, the Jupyter-based code must be rearranged and coding must be done separately.

Usually, Jupyter is used for testing to understand the operation of development modules interactively during the development process, visualization of experiment results, and creation of hands-on tutorials for education of developed libraries (as a result, effective for interaction with users).

This document aims to provide guidelines for developing machine learning projects with server tasks in mind. For ease of access, I will give an example with a very simple model for the MNIST data.

## Installation
* Step 1. git clone https://github.com/ro-ko/ml-experiment-template.git
* Step 2. open files with vscode
* Step 3. install dependent library for `ml-experiment-template`(conda activate before install requirements)
```shell
pip install -r requirements.txt
```





## Project structure

```shell
├── README.md
├── datasets                        # datasets directory
│   └── MNIST
├── sbin
│   └── generate_pipreqs.sh
└── src                             # source codes directory
    ├── data.py                     # datasets (DataSet, DataLoader) related script
    ├── experiments                 # experiment scripts directory
    │   ├── __init__.py
    │   ├── checkpoint
    │   ├── config
    │   │   └── default.yml
    │   ├── exp_hyper_param.py
    │   └── log
    ├── main.py                     # user input script
    ├── models                      # model codes (suppose that there are several models)
    │   ├── __init__.py
    │   └── mymodel                 # 'mymodel'codes
    │       ├── __init__.py
    │       ├── eval.py             # mymodel evaluation with test data (estimate accuracy)
    │       ├── model.py            # mymodel implementation (forward function)
    │       └── train.py            # mymodel train with data, hyper_param (gradient descent & backprop)
    ├── test.py
    └── utils.py
```

## How to call this in a terminal?
`ml-experiment-template` can experiment using script arguments with configuration file(.yml) in terminal.

Implemented by [fire](https://github.com/google/python-fire)

1. Make configuration file(.yml) in `src/experiments/config` [[yaml syntax](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)]
2. Open terminal and change directory to `src`.
3. Run script code like as below.

```
python -m main --cfg_name default
or
python main.py --cfg_name=default
```
## Additional information for template
You can check argument explaination by `python -m main --help`
It prints result from main.py -> main() function comment.
```
NAME
    main.py - Handle user arguments of ml-project-template

SYNOPSIS
    main.py <flags>

DESCRIPTION
    Handle user arguments of ml-project-template

FLAGS
    -c, --cfg_name=CFG_NAME
        Default: 'default'
        config file(.yml) name
```

And add additional terminal shell and run tensorboard then you can check several result about experiment.  
Tensorboard code is in `train.py`.  
```
tensorboard --logdir runs
```  
`runs` is default directory of tenrsorboard logging, you can log your specified directory. [[pytorch tensorboard tutorial](https://pytorch.org/docs/stable/tensorboard.html)] [[kor-docs version](https://tutorials.pytorch.kr/recipes/recipes/tensorboard_with_pytorch.html)]  

Per 5 epochs, model.pt is saved in `checkpoint` directory and it's code in `train.py`. Too short epoch term requires many storage resource if model is heavy, so you need to adjust proper value.  

`log` is directory which saves additional informatio for experiment and metric value. It use `logger` library in python and `loguru` useful tool for logging.  

As above, if shell commands can be processed on the terminal, the following can be used.
* Parallel experimentation on various parameters becomes easy (if a shell command is called in the background several times within the available CPU range, it is automatically scheduled and executed in parallel by the operating system)
  - Experiment results should be stored separately on disk.
  - Too many process calls slow down due to scheduling overhead (so the number of instructions executed simultaneously must be well controlled)
* `tmux` is a tool that opens multiple terminals on one screen in Linux and allows them to be executed individually. In tmux, sessions are maintained unless the user turns it off or the server goes down.So, if you use tmux after connecting to the server, you can perform experiments stably without writing a shell script for background call.(the reason why this is necessary is that the server connection is automatically disconnected when the client terminal is closed.)
  - Usually, tmux is used a lot when you need to experiment in a hurry or when it is annoying to do shell scripting.

 

## How to add a new model?
add file like `mymodel` type in `models` directory.

## Todo list
* [ ] Write 'how to connect with neptuneAI or WandDB?'


