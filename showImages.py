
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageShower:
  
  def __init__(self, file):
    self.imgs = []

    self.imgNames = [file for file in glob.glob(file+"/*.jpg")]
    for name in self.imgNames[:10]:
      self.imgs.append( mpimg.imread(name))
    
  def imageNames(self):
    return self.imgNames

  def show(self):
    plt.figure(figsize=(50, 50))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(self.imgs[i])
        plt.title('Image %d' % i, fontsize=10)
        plt.axis("off")
    plt.tight_layout()

    plt.show()
