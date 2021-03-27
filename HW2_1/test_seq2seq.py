#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This is part of the code for CPSC8830-Deep Learning HW02  Video caption generation.
    @author: Sadegh Sadeghi Tabas
    Email: sadeghs@clemson.edu 
    
    Created on Mon Mar 08 21:04:44 2021
    
    test_seq2seq.py: Model testing 
"""

# import libraries
import sys
import torch
import json
from train_seq2seq import training, dictionary, test_data
from torch.utils.data import DataLoader
from bleu_eval import BLEU


if not torch.cuda.is_available():
    model = torch.load('SavedModel/model4.h5', map_location=lambda storage, loc: storage)
else:
    model = torch.load('SavedModel/model4.h5')

dataset = test_data('{}/testing_data/feat'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
training_json = 'MLDS_hw2_1_data/training_label.json'
helper = dictionary(training_json, min_word_count=3)
testing = training(model=model, test_dataloader=testing_loader, helper=helper)
for epoch in range(1):
    ss = testing.test()

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

# Bleu Eval
test = json.load(open('{}/testing_label.json'.format(sys.argv[1]),'r'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))
