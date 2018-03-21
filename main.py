from __future__ import absolute_import
from __future__ import print_function
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import os

from os.path import join as pjoin
import h5py
from RAKE import rake
import sys
from urlFinder import UrlFinder
import imgdownloader
import tfidf
from showImages import ImageShower
from tfidf_word_freq import TfIdfWordFreq
from creation_resnet_rpz import Resnet
from explore import Explorer
import matplotlib.image as mpimg

stop_words = []
stop_words += rake.load_stop_words('SmartStoplist.txt')
stop_words += rake.load_stop_words('FoxStoplist.txt')
stop_words += stopwords.words('english')



file = input("Name of file ? ") or 'texts/ocean'
print("\nWorking on file: " + file)
text = open(file, 'r').read()

methodeKeywords = input('\nQuel algorithme utiliser ? (tfidf/wordfreq/rake) ')

if methodeKeywords == 'wordfreq':
  wordFreq = TfIdfWordFreq(stop_words)
  keywords = wordFreq.keywords(text)

elif methodeKeywords != 'rake':
  print("\nLancement de l'algorithme Tf-Idf")
  tfidf = tfidf.TfIdf(stop_words)
  keywords = tfidf.keywords(text)

else:

  # 1. initialize RAKE by providing a path to a stopwords file and setting phrase length in words to 1
  stoppath = "RAKE/FoxStoplist.txt"
  rake_object = rake.Rake(stop_words, 2, 1, 2)

  # 2. run on RAKE on a given text
  keywords = rake_object.run(text)
  keywords = keywords[:5]


print("\nKeywords obtenus :")
for word, score in keywords:
  print("\t", word, ":", score)


find = input("\nFind images ? (y/n) ")
if find == 'y':
  words = []
  for word in keywords:
    words.append(word[0])

  urlFinder = UrlFinder()
  img_urls = urlFinder.findFromWords(words)

  print("\n \t Urls found:")
  print(img_urls)

else:
  img_urls=[
    (wn.synset('ocean.n.01'), [
      'http://farm4.static.flickr.com/3041/2888019312_da1509ce3d.jpg', 'http://farm1.static.flickr.com/162/342921378_23f49f2701.jpg', 'http://farm1.static.flickr.com/47/110891522_1c88cd6bce.jpg', 'http://farm4.static.flickr.com/3409/3198834261_2640fbd259.jpg', 'http://farm1.static.flickr.com/204/470286018_f0966ead25.jpg', 'http://farm1.static.flickr.com/53/119972596_b9278c2454.jpg', 'http://farm4.static.flickr.com/3052/2616079309_cf30f554f9.jpg', 'http://farm1.static.flickr.com/242/453208551_42208a9253.jpg', 'http://farm4.static.flickr.com/3157/2909074563_0a972ae6ff.jpg', 'http://farm3.static.flickr.com/2640/4107071309_274c5efd22.jpg'
      ]),
    (wn.synset('water.n.06'), [
      'http://farm1.static.flickr.com/50/141618794_0b30839512.jpg', 'http://farm4.static.flickr.com/3240/2842144940_d50bc27d49.jpg', 'http://farm3.static.flickr.com/2204/1808115684_8506dcd679.jpg', 'http://farm4.static.flickr.com/3285/3061137477_b579520c8f.jpg', 'http://farm4.static.flickr.com/3410/3329791790_7319591ddc.jpg', 'http://farm3.static.flickr.com/2136/3534403642_eb8c64f77a.jpg', 'http://farm3.static.flickr.com/2053/2211522535_0c090f861e.jpg', 'http://farm3.static.flickr.com/2575/3881891825_5f995810f9.jpg', 'http://farm4.static.flickr.com/3371/3414586241_462262794b.jpg', 'http://farm3.static.flickr.com/2486/4133911360_7db5e237d1.jpg'
      ]),
    (wn.synset('airfoil.n.01'), ['http://www.airstrike.com.au/images/Product/Vladimir%20Models/g2en5001.jpg', 'http://www.nesail.com/pictures/impulse2.jpg', 'http://farm4.static.flickr.com/3166/2699711511_34d7c7cdb7.jpg', 'http://www.se2funworks.com/photogallery/EF%20Extra/EF%20300%20Thumbnail.JPG', 'http://farm2.static.flickr.com/1416/534259053_4b70a3460b.jpg', 'http://www.themarcs.org/uploads/images/traders/airfoilz_yak.jpg', 'http://farm4.static.flickr.com/3028/3020546041_80ff5b23c6.jpg', 'http://site.nitroplanes.com/biplane00517.jpg', 'http://farm4.static.flickr.com/3078/2569442912_79a7cda124.jpg', 'http://farm4.static.flickr.com/3228/2445842255_9068949398.jpg'
    ])
  ]


down = input("\nDownload images ? (y/n) ")
if down == 'y':
  urls = []
  for synsetTuple in img_urls:
    urls += synsetTuple[1]

  downloader = imgdownloader.ImageNetDownloader()
  downloader.downloadImagesByURLs("0", urls, 100)

  print("\n\tDone")


explorer = Explorer(img_urls)
explorer.explore()





# resnet = Resnet()
# X = resnet.process("images/ocean")

f = h5py.File('images/ocean/sorted_aug_img_emb.h5', 'r')
X = f['img_emb']

pca = PCA(n_components=10)
X_r = pca.fit(X).transform(X)
print("shape:", X_r.shape)

kmeans = KMeans(n_clusters=5)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(X_r)

LABEL_COLOR_MAP = {0 : 'r', 1 : 'g', 2 : 'b', 3 : 'y', 4 : 'k',}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]


list_files = [ f for f in sorted(os.listdir("images/ocean"))
  if (os.path.isfile(pjoin("images/ocean", f)) and f[-4:] == '.jpg')]

print(len(list_files),len(X_clustered))
files_clusters=zip(list_files,X_clustered)

clusters=[[],[],[],[],[]]
for t in files_clusters:
  clusters[t[1]].append(t[0])


imgs = []

for i in range(5):
  for name in clusters[i][:10]:
    imgs.append(mpimg.imread("images/ocean/"+name))

plt.figure(figsize=(50, 50))
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(imgs[i])
    # plt.title('Image %d' % i, fontsize=10)
    plt.axis("off")
plt.tight_layout()

plt.show()



# fig, ax = plt.subplots()
# ax.scatter(X_r[:,2], X_r[:,3], c= label_color, alpha=0.5)
# # n = range(27)
# # for i, txt in enumerate(n):
# #     ax.annotate(txt, (X_r[i,0], X_r[i,1]))

# plt.show()
