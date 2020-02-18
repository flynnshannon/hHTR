from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess

class Sample:
  "element of dataset"
  def __init__(self, gtText, filePath):
    self.gtText = gtText
    self.filePath = filePath

class Batch:
  "batch of images with ground truth texts"
  def __init__(self, gtTexts, imgs):
    self.imgs = np.stack(imgs, axis = 0)
    self.gtTexts = gtTexts

class DataLoader:
  def __init__(self, filePath, batchSize, imgSize, maxTextLen):
    "loader for dataset at given location, preprocess images and text"
    assert filePath[-1]=='/'

    self.dataAugmentation = False
    self.currIdx = 0
    self.batchSize = batchSize
    self.samples = []
    self.imgSize = imgSize
    self.maxTextLen = maxTextLen

    #fix later
    f = open('words.txt')
    chars = set()
    bad_samples = []
    bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
    for line in f:
      if not line or line[0] == '#':
        continue
        #ignore comments
      lineSplit = line.strip().split(' ')
      assert len(lineSplit) >= 9

      fileNameSplit = lineSplit[0].split('-')
      fileName = filePath + 'words/' + fileNameSplit[0] + '/'+fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

      #ground truth text columns starting at 9
      gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
      chars = chars.union(set(list(gtText)))

      #check that image not empty
      if not os.path.getsize(fileName):
        bad_samples.append(lineSplit[0] + '.png')
        continue

      #put sample into list
      self.samples.append(Sample(gtText, fileName))

    #some IAM images are damaged
    if set(bad_samples) != set(bad_samples_reference):
      print('Warning, damaged lines found: ', bad_samples)
      print('Damaged images expected: ', bad_samples_reference)

    #split into training and validation sets - 95:5
    splitIdx = int(0.95 * len(self.samples))
    self.trainSamples = self.samples[:splitIdx]
    self.validationSamples = self.samples[splitIdx:]

    #put words into lists
    self.trainWords = [x.gtText for x in self.trainSamples]
    self.validationWords = [x.gtText for x in self.validationSamples]

    #samples per epoch
    self.numTrainSamplesPerEpoch = 25000

    self.trainSet()
    self.charList = sorted(list(chars))

  def truncateLabel(self, text, maxTextLen):
    #ctc_loss can't compute loss if it cant map textlabel to input labels
    #repeat letters cost double because of the blank symbol needing to be inserted
    #if too long label is provided ctc_loss returns infinite gradient
    cost = 0
    for i in range(len(text)):
      if i != 0 and text[i] == text[i-1]:
        cost += 2
      else:
        cost += 1
      if cost > maxTextLen:
        return text[:1]
    return text

  def trainSet(self):
    #randomly chosen sunset of training sets
    self.dataAugmentation = True
    self.currIdx = 0
    random.shuffle(self.trainSamples)
    self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

  def validationSet(self):
    #switch to validation
    self.dataAugmentation = False
    self.currIdx = 0
    self.samples = self.validationSamples

  def getIteratorInfo(self):
    return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

  def hasNext(self):
    return self.currIdx + self.batchSize <= len(self.samples)

  def getNext(self):
    batchRange = range(self.currIdx, self.currIdx + self.batchSize)
    gtTexts = [self.samples[i].gtText for i in batchRange]
    imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
    self.currIdx += self.batchSize
    return Batch(gtTexts, imgs)
