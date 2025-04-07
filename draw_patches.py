import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import cv2 as cv

from scipy.ndimage import label

os.chdir('C://Users/researchlabs/Documents/Brown/Maiasaura')

subjects = pd.read_excel('subject_lookup.xlsx')
# n = len(subjects)


##k =  1
##os.chdir(r"D:\Check Lab\Projects\OSU-CHS Projects\2018-05 Maiasaura Humeri\New Project Code Images")
##
##a = cv.imread(f'./{subjects.Subject[k]}/{subjects.Path[k]}')
##
##b = 1
##sil = 1
##direc = ['North', 'South', 'East', 'West']
##for d in direc:
##
##    
##    shift_x = subjects[f'{d}_x'][k]
##    shift_y = subjects[f'{d}_y'][k]
##    # shift_x = 0
##    # shift_y = 0
##
##    if np.isnan(shift_x) |  np.isnan(shift_y):
##        continue
##    shift_x = int(shift_x)
##    shift_y = int(shift_y)
##    print(d)
##    
##    if type(b) == int:
##        b = 255 - cv.imread(f'./{subjects.Subject[k]}/{d} Secondary.jpg')
##        b = np.roll(b, shift_x, axis=0)
##        b = np.roll(b, shift_y, axis=1)
##
##        sil = 255 - cv.imread(f'./{subjects.Subject[k]}/{d} Section.jpg')
##        # sil = np.roll(sil, shift_x, axis=0)
##        # sil = np.roll(sil, shift_y, axis=1)
##    else:
##        c = 255 - cv.imread(f'./{subjects.Subject[k]}/{d} Secondary.jpg')
##        c = np.roll(c, shift_x, axis=0)
##        c = np.roll(c, shift_y, axis=1)
##        
##        b = b + c
##
##        testy = 255 - cv.imread(f'./{subjects.Subject[k]}/{d} Section.jpg')
##        # testy = np.roll(testy, shift_x, axis=0)
##        # testy = np.roll(testy, shift_y, axis=1)
##        sil = sil + testy
##
##
##a = cv.resize(a, [b.shape[1], b.shape[0]])
##
##b = np.sum(b, axis=2)
##
##shift_x = 0
##shift_y = 0
##b = np.roll(b, shift_x, axis=0)
##b = np.roll(b, shift_y, axis=1)
##
##
##print(b.shape)
##
##b = b > 10
##b = b.astype(float)
##print(np.sum(b))
##
##sil = np.sum(sil, axis=2)
##sil = sil > 10
##sil = sil.astype(float)
##
##
###plt.imshow(a)
##plt.figure(figsize=(4, 4))
##plt.imshow(a)
##plt.imshow(b, cmap='gray', alpha=0.5)
##plt.imshow(sil, cmap='gray', alpha=0.25)
##plt.show()








# Actually drawing patches. 
k =  0
data = np.load(f'./data/train_data.npy')
masks = np.load(f'./data/train_masks.npy')

for w in range(10):
    fig, ax = plt.subplots(4, 6, figsize=(10, 10))
    
    for i in range(6):

        ax[0, i].imshow(masks[k+i+w*6, :, :] > 0, cmap='gray')
        for j in range(3):
            ax[j+1, i].imshow(data[k+i+w*6, :, :, j], cmap='plasma')

        print(np.sum(masks[k+i+w*6, :, :]))

    # Testing circularity:
    ##    contours, _ = cv.findContours(masks[k, :, :].astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ##    area = cv.contourArea(contours[0])
    ##    perim = cv.arcLength(contours[0], True)
    ##
    ##    print(4*np.pi * area / (perim**2))

    for a in ax.flatten():
        a.axis('off')
        
    plt.show()

