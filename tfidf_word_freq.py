import operator
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log

from wordfreq import word_frequency


class TfIdfWordFreq:
  
  def __init__(self, stopwords):
    self.stopwords = stopwords
    
    self.N = 1024908267229
    f = open('count_words.txt')
    data = f.readlines()
    lines = []
    for line in data:
      l = line.split()
      l[1] = int(l[1])/self.N
      lines.append(l)
    self.wordsFreq = np.array(lines)

  def computeWordFreq(self, text):
    filtered_words = [word for word in text.split() if word not in self.stopwords and word != "."]
    words_vector = []
    words_freq = []
    for word in filtered_words:
      if word in words_vector:
        words_freq[words_vector.index(word)] += 1
      else:
        words_vector.append(word)
        words_freq.append(1)
    return list(zip(words_vector,words_freq))

  def keywords(self, text, numb_to_retain = 5):
    text= text.lower()
    wordFreqVector = self.computeWordFreq(text)
    wordScores = []
    for wordTuple in wordFreqVector:
      word = wordTuple[0]
      wordTextFreq = wordTuple[1]
      wordFreq = word_frequency(word, 'en', minimum=1e-7)
      
      # i = np.where(self.wordsFreq==word)[0]
      # if i.size == 0:
      #   wordFreq = 1/self.N
      # else:
      #   wordFreq = float(self.wordsFreq[i,1][0])
      # wordScore = wordTextFreq * (log(1/wordFreq) + 1)
      wordScore = wordTextFreq * (log(1/wordFreq) + 1)
      wordScores.append((word,wordScore))
    
    wordScores = sorted(wordScores, key=operator.itemgetter(1), reverse=True)
    numb_to_retain = 10
    keywords = [my_tuple for my_tuple in wordScores[:numb_to_retain]]

    return keywords
    
