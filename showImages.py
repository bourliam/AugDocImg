
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle


class ImageShower:
  
  def __init__(self, files):

    self.imgNames = glob.glob(files+"/*.jpg")
    
    
  def imageNames(self):
    return self.imgNames[:10]

  def show(self):
    
    shuffle(self.imgNames)

    imgs = []

    for name in self.imgNames[:10]:
      imgs.append(mpimg.imread(name))

    plt.figure(figsize=(50, 50))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(imgs[i])
        plt.title('Image %d' % i, fontsize=10)
        plt.axis("off")
    plt.tight_layout()

    plt.show()
