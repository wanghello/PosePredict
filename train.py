#################################################     
#     Filename: train.py
#       Author: wang - shawn_wang@163.com
#  Description: ---
#       Create: 2019-12-19 22:39:25
#Last Modified: 2019-12-19 22:39:25
#################################################     
#!/usr/bin/env python
# coding=utf-8

from dataset import *

#init dataset
BATCH_SIZE = 4
cross_validation = False

#init model
model = ''

def train_model(epoch, history=None):
    model.train()
    train_loader, dev_loader = load_dataset(cross_validation, BATCH_SIZE)

