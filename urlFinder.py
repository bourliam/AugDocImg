
import sys
import os
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from random import shuffle
import requests
import time


class UrlFinder:
    def __init__(self):
        self.synset_list = open('synset_list.txt', 'r').read()
        
    def getUrl(self, wnid):
        format_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'
        return format_url.format(wnid)

    def getResult(self, syn_id, num_urls=10):
        url = self.getUrl(syn_id)
        r = requests.get(url)
        urls = r.text.splitlines()
        shuffle(urls)

        if urls[0] == 'The synset is not ready yet. Please stay tuned!':
            print("Pas d'images !")
            urls.pop()


        return urls[:num_urls]

    def searchUrls(self, synset, num_urls=10):
        
        urls = []
        syn_id = self.wnid(synset)

        if self.synset_list.find(syn_id) == -1 :
            print(synset.name(), "pas dans la liste")
            return

        print("Searching urls for " + synset.name())
        
        try:
            urls = self.getResult(syn_id, num_urls)
        except (ValueError, requests.exceptions.RequestException):
            return # ok, never mind - try a different synset

        return (synset, urls)

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

    def appendIfExist(self, list1, list2):
        if list2 == None:
            return list1
        else:
            return list1.append(list2)


    def findInHyponyms(self,synsets,max_imgs=100):
        img_urls=[]
        for synset in synsets:
            for hn in synset.hyponyms():
                if len(img_urls) >= max_imgs:
                    break
                self.appendIfExist(img_urls, self.searchUrls(hn))
        return img_urls



    def findFromSynsets(self, synsets, hyponyms = False, max_imgs = 100):
        img_urls=[]
       
        print("\nSynsets: ", synsets)

        for synset in synsets:
            if len(img_urls) >= max_imgs:
                break
            self.appendIfExist(img_urls, self.searchUrls(synset))

        # Get hyponyms
        if hyponyms:
            img_urls += self.findInHyponyms(synsets)

        return img_urls
    


    def findWithHypernyms(self,synsets,max_imgs=100):

        img_urls = self.findFromSynsets(synsets)

        if len(img_urls) == 0:
            for synset in synsets:
                for hypernym in synset.hypernyms():
                    self.appendIfExist(img_urls, self.searchUrls(hypernym))
                    if len(img_urls) > max_imgs:
                        break
        return img_urls
    


    def findFromWords(self, words, max_imgs = 100):
        synsets=[]
        for word in words:
            wnl = WordNetLemmatizer()
            lem = wnl.lemmatize(word, pos=wn.NOUN)

            synsets += wn.synsets(lem, pos=wn.NOUN)
        synsets= list(set(synsets))
        return self.findWithHypernyms(synsets, max_imgs)
