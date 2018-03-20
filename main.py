from __future__ import absolute_import
from __future__ import print_function
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from RAKE import rake
import sys
import urlFinder
import imgdownloader
import tfidf
from showImages import ImageShower
from tfidf_word_freq import TfIdfWordFreq
from creation_resnet_rpz import Resnet

stop_words = []
stop_words += rake.load_stop_words('SmartStoplist.txt')
stop_words += rake.load_stop_words('FoxStoplist.txt')
stop_words += stopwords.words('english')

file = input("Name of file ? ") or 'texts/ocean'
print("Working on file: " + file)
text = open(file, 'r').read()

methodeKeywords = input('Quel algorithme utiliser ? (tfidf/wordfreq/rake) ')

if methodeKeywords == 'wordfreq':
  wordFreq = TfIdfWordFreq(stop_words)
  keywords = wordFreq.keywords(text)

elif methodeKeywords != 'rake':
  print("Lancement de l'algorithme Tf-Idf")
  tfidf = tfidf.TfIdf(stop_words)
  keywords = tfidf.keywords(text)

else:

  # 1. initialize RAKE by providing a path to a stopwords file and setting phrase length in words to 1
  stoppath = "RAKE/FoxStoplist.txt"
  rake_object = rake.Rake(stop_words, 2, 1, 2)

  # 2. run on RAKE on a given text
  keywords = rake_object.run(text)
  keywords = keywords[:5]

print()
print("Keywords obtenus :")
for word, score in keywords:
  print("\t", word, ":", score)
print()

find = input("Find images ? (y/n) ")
if find == 'y':
  urlFinder = urlFinder.urlFinder()
  img_urls = []
  for word in keywords:
    img_urls = urlFinder.find(word[0])
  print(img_urls)

down = input("Download images ? (y/n) ")
if down == 'y':
  urls = []
  for synsetTuple in img_urls:
    urls += synsetTuple[1]

  downloader = imgdownloader.ImageNetDownloader()
  downloader.downloadImagesByURLs("report", urls, 100)

  print("Done")

print("Look at thoses images and note your favorite's number")
imgShower = ImageShower("report/report_images")
imgShower.show()
imgNames = imgShower.imageNames()

image = int(input("What was the number ? "))
print("The name of the file is: ")
print(imgNames[image])

resnet = Resnet()
X = resnet.process("images/images_images")

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print("shape:", X_r.shape)



fig, ax = plt.subplots()
ax.scatter(X_r[:,0], X_r[:,1])
n = range(27)
for i, txt in enumerate(n):
    ax.annotate(txt, (X_r[i,0], X_r[i,1]))

plt.show()
