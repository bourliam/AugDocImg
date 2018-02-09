
import sys
import os
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import requests
import time


class urlFinder:
    def __init__(self):
        self.img_urls = []
        self.synset_list = open('synset_list.txt', 'r').read()
        
    def getUrl(self, wnid):
        format_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'
        return format_url.format(wnid)

    def getResult(self, syn_id):
        url = self.getUrl(syn_id)
        r = requests.get(url)
        urls = r.text.splitlines()

        if urls[0] == 'The synset is not ready yet. Please stay tuned!':
            print("Oups pas d'images !")
            urls.pop()

        return urls[:10]

    def searchUrls(self, synset):
        
        urls = []
        syn_id = self.wnid(synset)

        if self.synset_list.find(syn_id) == -1 :
            print(synset.name(), "pas dans la liste")
            return urls

        print("Searching urls for " + synset.name())
        
        try:
            urls += self.getResult(syn_id)
        except (ValueError, requests.exceptions.RequestException):
            return # ok, never mind - try a different synset

        return urls

    def get_hyponyms(self, synset, depth=0):
        hyponyms = set()
        depth += 1
        if depth == 40: # avoid maximum recursion exceeded errors
            return set()
        for hyponym in synset.hyponyms():
            hyponyms |= set(self.get_hyponyms(hyponym, depth))
        return hyponyms | set(synset.hyponyms())

    def wnid(self, synset):
        return '%s%.8d' % (synset.pos(), synset.offset())


    def find(self, word,  max_imgs = 100):
        img_urls = self.img_urls
       

        wnl = WordNetLemmatizer()
        lem = wnl.lemmatize(word, pos=wn.NOUN)

        synsets = wn.synsets(lem, pos=wn.NOUN)

        print("Synsets: ", synsets)

        for synset in synsets:
            if len(img_urls) >= max_imgs:
                break
            img_urls += self.searchUrls(synset)

        # 2. Get hyponyms to sample from
        if len(img_urls) == 0:
            for synset in synsets:
                for hn in self.get_hyponyms(synset):
                    if len(img_urls) >= max_imgs:
                        break
                    img_urls += self.searchUrls(hn)
                

        # 3. If no images, try hyponyms of hypernyms
        if len(img_urls) == 0:
            for synset in synsets:
                for hypernym in synset.hypernyms():
                    img_urls += self.searchUrls(hypernym)
                    for hn in self.get_hyponyms(hypernym):
                        img_urls += self.searchUrls(hn)
                        if len(img_urls) > max_imgs:
                            break

        print('Got ', len(img_urls), 'urls!')

        return img_urls



