from __future__ import absolute_import
from __future__ import print_function
from nltk.corpus import stopwords
from RAKE import rake
import sys
import urlFinder
import imgdownloader
import tfidf


stop_words = []
stop_words += rake.load_stop_words('SmartStoplist.txt')
stop_words += rake.load_stop_words('FoxStoplist.txt')
stop_words += stopwords.words('english')

file = input("Name of file ? ") or 'texts/ocean'
print("Working on file: " + file)
text = open(file, 'r').read()

methodeKeywords = input('Quel algorithme utiliser ? (tfidf/rake) ')

if methodeKeywords != 'rake':
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
if find != 'y':
  exit()

urlFinder = urlFinder.urlFinder()
img_urls = []
for word in keywords:
  img_urls = urlFinder.find(word[0])
print(img_urls)
down = input("Download images ? (y/n) ")
if down != 'y':
  exit()

urls = []
for synsetTuple in img_urls:
  urls += synsetTuple[1]

downloader = imgdownloader.ImageNetDownloader()
downloader.downloadImagesByURLs("images", urls, 100)

print("Done")
