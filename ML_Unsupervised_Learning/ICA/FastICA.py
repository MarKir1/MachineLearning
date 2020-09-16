### http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICA09.pdf
### from __future__ import division
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def process(images, new_images):
    # TODO: fix means and variances of images

    ### axis denotes axes along which mean & std reductions are to be performed

    mean1 = np.mean(images)
    mean2 = np.mean(new_images)
    std1 = np.std(images)
    std2 = np.std(new_images)
    c = std1 / std2
    
    new_images -= mean2
    new_images = (new_images * c + mean1) + mean2
    
    return new_images

if __name__ == '__main__':
    images = np.array([mpimg.imread(f'gen_img{i}.png').flatten() for i in range(5)])
    # TODO: Fit Fast ICA and get new images

    new_images = None
    ica = FastICA(n_components = 3, whiten = True)
    new_images = ica.fit_transform(images.T)
    new_images = new_images.T
    new_images = new_images * np.array([-1,1,1])[None].T
    new_images = new_images.reshape((3, 800, 600, 4))
    new_images = process(images, new_images)
    

    for image in new_images: 
        plt.imshow(image.clip(min=0, max=1))
        plt.show()
        
        
        