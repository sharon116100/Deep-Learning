import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
from matplotlib import animation
import cv2
from itertools import cycle

##Write down your visualization code here

## Animation for your generation
##input : image_list (size = (the number of sample times, how many samples created each time, image )   )
#img_list = []
videoWriter = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1058,530))

fig = plt.figure(figsize=(8,8))
ims = []
img_names = glob.glob('./img_GAN'+'/*100.png')
for img_name in img_names:
#    img = Image.open(img_name).convert('RGB')
#    img_list.append(img)
#    img = np.array(mgimg.imread(img_name))
    img = cv2.imread(img_name)
    imgplot = plt.imshow(img)
    ims.append([imgplot])
    cv2.imshow('frame', img)
    cv2.waitKey(30)

#plt.axis("off")
#ims = [[plt.imshow(i, animated=True)] for i in img_list]
#ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

#HTML(ani.to_jshtml())

#ani.save("animation.gif", writer='imagemagick')
#plt.show()
# https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.html#matplotlib.animation.Animation.save
#key = 0
#while key & 0xFF != 27:
#    cv2.imshow('window title', next(img_iter))
#    key = cv2.waitKey(1000)