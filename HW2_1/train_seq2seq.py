#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This is part of the code for CPSC8830-Deep Learning HW02  Video caption generation.
    @author: Sadegh Sadeghi Tabas
    Email: sadeghs@clemson.edu 
    
    Created on Mon Mar 08 21:04:44 2021
    
    train_seq2seq.py: Model training 
"""

# import libraries
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import random
from model import MODELS, encoderRNN, decoderRNN
import datetime
import sys
import os
import time
import json
import re


class dictionary(object):
    """
    Helper class for processing words
    """

    def __init__(self, filepath, min_word_count=10):

        # define public variables
        self.filepath = filepath
        self.min_word_count = min_word_count

        # define private variables
        self._word_count = {}
        self.vocab_size = None
        self._good_words = None
        self._bad_words = None
        self.i2w = None
        self.w2i = None

        # initialize class
        self._initialize()
        self._build_mapping()
        self._sanitycheck()


    def _initialize(self):
        with open(self.filepath, 'r') as f:
            file = json.load(f)

        for d in file:
            for s in d['caption']:
                word_sentence = re.sub('[.!,;?]]', ' ', s).split()

                for word in word_sentence:
                    word = word.replace('.', '') if '.' in word else word
                    self._word_count[word] = self._word_count.get(word, 0) + 1

        bad_words = [k for k, v in self._word_count.items() if v <= self.min_word_count]
        vocab = [k for k, v in self._word_count.items() if v > self.min_word_count]

        self._bad_words = bad_words
        self._good_words = vocab

    def _build_mapping(self):
        #dictionaries for mapping word to index and vice versa
        useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
        self.i2w = {i + len(useful_tokens): w for i, w in enumerate(self._good_words)}
        self.w2i = {w: i + len(useful_tokens) for i, w in enumerate(self._good_words)}
        for token, index in useful_tokens:
            self.i2w[index] = token
            self.w2i[token] = index

        self.vocab_size = len(self.i2w) + len(useful_tokens)

    def _sanitycheck(self):
        attrs = ['vocab_size', '_good_words', '_bad_words', 'i2w', 'w2i']
        for att in attrs:
            if getattr(self, att) is None:
                raise NotImplementedError('Class {} has an attribute "{}" which cannot be None. Error location: {}'.format(__class__.__name__, att, __name__))

    def reannotate(self, sentence):
        #replaces word with <UNK> if word is infrequent
        sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        sentence = ['<SOS>'] + [w if (self._word_count.get(w, 0) > self.min_word_count) \
                                    else '<UNK>' for w in sentence] + ['<EOS>']
        return sentence

    def word2index(self, w):
        return self.w2i[w]
    def index2word(self, i):
        return self.i2w[i]
    def sentence2index(self, sentence):
        return [self.w2i[w] for w in sentence]
    def index2sentence(self, index_seq):
        return [self.i2w[int(i)] for i in index_seq]
    
    
class LossFun(nn.Module):
    def __init__(self):
        super(LossFun, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = 0
        self.avg_loss = None

    def forward(self, x, y, lengths):
        # first dim of x and y is the same (equals to batch size)
        batch_size = len(x)
        predict_cat = None
        groundT_cat = None
        flag = True

        for batch in range(batch_size):
            predict      = x[batch]
            ground_truth = y[batch]
            seq_len = lengths[batch] -1

            predict = predict[:seq_len]
            ground_truth = ground_truth[:seq_len]
            if flag:
                predict_cat = predict
                groundT_cat = ground_truth
                flag = False
            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

        try:
            assert len(predict_cat) == len(groundT_cat)

        except AssertionError as error:
            print('prediction length is not same as ground truth length')
            print('prediction length: {}, ground truth length: {}'.format(len(predict_cat), len(groundT_cat)))

        self.loss = self.loss_fn(predict_cat, groundT_cat)
        self.avg_loss = self.loss/batch_size
        
        return self.loss
    
    
    
class training_data(Dataset):
    def __init__(self, label_json, training_data_path, helper, load_into_ram=False):
        # check if file path exists
        if not os.path.exists(label_json):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(label_json, __name__))
        if not os.path.exists(training_data_path):
            raise FileNotFoundError('File path {} does not exist. Error location: {}'.format(training_data_path, __name__))


        self.training_data_path = training_data_path
        # format (avi id, corresponding sentence)
        self.data_pair = []
        self.load_into_ram = load_into_ram
        self.helper = helper


        with open(label_json, 'r') as f:
            label = json.load(f)
        for d in label:
            for s in d['caption']:
                s = self.helper.reannotate(s)
                s = self.helper.sentence2index(s)
                self.data_pair.append((d['id'], s))

        if load_into_ram:
            self.avi = {}

            files = os.listdir(training_data_path)

            for file in files:
                key = file.split('.npy')[0]
                value = np.load(os.path.join(training_data_path, file))
                self.avi[key] = value


    def __len__(self):
        return len(self.data_pair)


    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        avi_file_path = os.path.join(self.training_data_path, '{}.npy'.format(avi_file_name))
        data = torch.Tensor(self.avi[avi_file_name]) if self.load_into_ram else torch.Tensor(np.load(avi_file_path))
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)
    

class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())

        return self.avi[idx]

# Create mini-batch tensors from the list of tuples (image, caption)
def minibatch(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

class training(object):
    def __init__(self, model, train_dataloader=None, test_dataloader=None, helper=None):
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

        # Use cuda is available
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
            print('GPU is available')
        else:
            self.model = model.cpu()

        # define hyper parameters
        self.parameters = model.parameters()
        self.loss_fn = LossFun()
        self.loss = None
        self.optimizer = optim.Adam(self.parameters, lr=0.001)
        self.helper = helper

    def train(self, epoch):
        self.model.train()

        test_avi, test_truth = None, None

        for batch_idx, batch in enumerate(self.train_loader):
            # prepare data
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # start training process
            self.optimizer.zero_grad()
            seq_logProb, seq_predictions = self.model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
            
            # eliminate <SOS>
            ground_truths = ground_truths[:, 1:]  
            loss = self.loss_fn(seq_logProb, ground_truths, lengths)
            loss.backward()
            self.optimizer.step()

            # print out training info
            if (batch_idx+1):
                info = self.get_training_info(epoch=epoch, batch_id=batch_idx, batch_size=len(lengths), total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader), loss=loss.item())
                print(info, end='\r')
                sys.stdout.write("\033[K")

        info = self.get_training_info(epoch=epoch, batch_id=batch_idx, batch_size=len(lengths), total_data_size=len(self.train_loader.dataset),
            n_batch=len(self.train_loader), loss=loss.item())
        print(info)
        
        # update loss for each epoch
        self.loss = loss.item()


    def eval(self):
        # set model to evaluation(testing) mode
        self.model.eval()
        test_predictions, test_truth = None, None
        for batch_idx, batch in enumerate(self.test_loader):
            # prepare data
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            # start inferencing process
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            ground_truths = ground_truths[:, 1:]
            test_predictions = seq_predictions[:3]
            test_truth = ground_truths[:3]
            break


    def test(self):
        
        # set model to evaluation(testing) mode
        self.model.eval()
        ss = []
        for batch_idx, batch in enumerate(self.test_loader):
            # prepare data
            id, avi_feats = batch
            if self.__CUDA__:
                avi_feats = avi_feats.cuda()
            id, avi_feats = id, Variable(avi_feats).float()

            # start inferencing process
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            test_predictions = seq_predictions
            result = [[x if x != '<UNK>' else 'something' for x in self.helper.index2sentence(s)] for s in test_predictions]
            result = [' '.join(s).split('<EOS>')[0] for s in result]
            rr = zip(id, result)
            for r in rr:
                ss.append(r)
        return ss

    def get_training_info(self,**kwargs):
        ep = kwargs.pop("epoch", None)
        bID = kwargs.pop("batch_id", None)
        bs = kwargs.pop("batch_size", None)
        tds = kwargs.pop("total_data_size", None)
        nb = kwargs.pop("n_batch", None)
        loss = kwargs.pop("loss", None)
        info = "Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(ep, (bID+1)*bs, tds, 100.*bID/nb, loss)
        return info


def main():
    # import data
    # import training and testing labels from associated json files
    training_json = 'MLDS_hw2_1_data/training_label.json'
    training_feats = 'MLDS_hw2_1_data/training_data/feat'
    testing_json = 'MLDS_hw2_1_data/testing_label.json'
    testing_feats = 'MLDS_hw2_1_data/testing_data/feat'

    helper = dictionary(training_json, min_word_count=3)
    train_dataset = training_data(label_json=training_json, training_data_path=training_feats, helper=helper, load_into_ram=True)
    test_dataset = training_data(label_json=testing_json, training_data_path=testing_feats, helper=helper, load_into_ram=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=minibatch)
    
    word_dim = 1024
    inputFeatDim = 4096
    hidden_size = 512
    dropout_percentage = 0.3
    output_dim = helper.vocab_size
    
    epochs_n = 100
    ModelSaveLoc = 'SavedModel'
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)

    encoder = encoderRNN(input_size=inputFeatDim, hidden_size=hidden_size, dropout_percentage=dropout_percentage)
    decoder = decoderRNN(hidden_size=hidden_size, output_size=output_dim, vocab_size=output_dim, word_dim=word_dim, dropout_percentage=dropout_percentage)
    model = MODELS(encoder=encoder, decoder=decoder)
    train = training(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, helper=helper)

    start = time.time()
    for epoch in range(epochs_n):
        train.train(epoch+1)
        train.eval()

    end = time.time()
    torch.save(model, "{}/{}.h5".format(ModelSaveLoc, 'model0'))
    print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))


if __name__ == '__main__':
    # Download dataset
    # pip install gdown
    # gdown 'https://drive.google.com/u/1/uc?export=download&confirm=B5wS&id=1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ'
    # tar -xf MLDS_hw2_1_data.tar.gz
    main()
    
    
    
    