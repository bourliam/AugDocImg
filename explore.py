

from showImages import ImageShower
from urlFinder import UrlFinder
from imgdownloader import ImageNetDownloader

class Explorer:
  
  def __init__(self, synsets_urls):
    self.isLeaf = False
    self.synsets_urls = synsets_urls

  def pickImage(self, imageFile):
    imgShower = ImageShower(imageFile)
    found = False
    while not found:
      print("\nLook at thoses images and note your favorite's number.")

      imgShower.show()
      imgNames = imgShower.imageNames()

      image = input("What was the number ? (if no good image leave blank) ")

      if image != '':
        image = int(image)
        found = True

    imgName = imgNames[image].split("/")[-1]
    print("\nThe name of the chosen file is:",imgName,"\n")

    return imgName

  def findSynsets(self, imageName):
    synsets = []
    for synsetTuple in self.synsets_urls:
      for url in synsetTuple[1]:
        if imageName == url.split("/")[-1] and synsetTuple[0] not in synsets:
          synsets.append(synsetTuple[0])
          break
    
    return synsets

  def downloadImages(self, img_urls, folderName=0):
    down = input("\nDownload images ? (y/n) ")
    print()
    if down == 'y':
      urls = []
      for synsetTuple in img_urls:
        urls += synsetTuple[1]

      downloader = ImageNetDownloader()
      downloader.downloadImagesByURLs(folderName, urls, 100)

      print("\nDone")


  def explore(self):
    i=0
    while not self.isLeaf:
      
      imageFile= "images/" + str(i)
      imgName = self.pickImage(imageFile)

      synsets = self.findSynsets(imgName)

      urlFinder = UrlFinder()
      self.synsets_urls = urlFinder.findFromSynsets(synsets, hyponyms = True)
      print("\n \t Urls found:")
      print(self.synsets_urls)

      i += 1
      self.downloadImages(self.synsets_urls, i)

      if(len(self.synsets_urls) == 1):
        self.isLeaf = True
      
    print('\n \n \tThanks, you can look in the folder "images/{}" to find plenty illustration images'.format(i))
      
