import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.model_selection import train_test_split

from scipy.ndimage import label
# from skimage.morphology import convex_hull_image

from tensorflow.image import resize #, pad_to_bounding_box

import albumentations as A

base_dir = r'C:\Users\researchlabs\Documents\Brown\Maiasaura'
data_dir = r'D:\Check Lab\Projects\OSU-CHS Projects\2018-05 Maiasaura Humeri\New Project Code Images'

os.chdir(base_dir)
subjects = pd.read_excel('subject_lookup.xlsx')
                         
os.chdir(data_dir)

#Global variables for run
n = 224

# patch_size = 112 # doubled after upsampling
patch_size = 112
patch_size_upsampled = n
# buffer = 16 # on either side.
#n = patch_size + 2*buffer # input to ML.

batch_size = 5
epochs = 10
increment = 8 # moved over from from_subject
direcs = ['North', 'South', 'East', 'West']


# subjects = ['H40', 'H40', 'H40']


##def add_buffer(input):
##    output = np.zeros((n, n))
##    output[buffer:(patch_size_upsampled + buffer), buffer:(patch_size_upsampled + buffer)] = input
##    return output


def get_mask(mask):
    # mask = zoom(mask, 2, order=1)
    mask = np.kron(mask, np.ones((2, 2))) # umsample 2x
    # mask = add_buffer(mask)

    # mask = cv.resize(mask, (224, 224), interpolation=cv.INTER_NEAREST)
    return mask


def get_patch(patch):
    r, g, b = cv.split(patch)
    # r = cv.GaussianBlur(r, (5, 5), 0)
    # g = cv.GaussianBlur(g, (5, 5), 0)
    # b = cv.GaussianBlur(b, (5, 5), 0)

    # no longer do histogram equilization here for ResNet.

##    r = add_buffer(r)
##    g = add_buffer(g)
##    b = add_buffer(b)

    #r = r / 255
    #g = g / 255
    #b = b / 255

    # ResNET normalization parameters:
##    r = (r - 123.68) / 58.393
##    g = (g - 116.779) / 57.12
##    b = (b - 103.939) / 57.375

    patch = np.stack([r, g, b], axis=2)
    
    # Upsample -> 224x224x3
    patch = resize(patch, (patch_size_upsampled, patch_size_upsampled), method='bilinear')

    # patch = pad(patch, [[buffer, buffer], [buffer, buffer], [0, 0]], mode='REFLECT')
    # patch = pad_to_bounding_box(patch, buffer, buffer, n, n)

    return patch
                       


##############################################
# Data augmentation pipeline
##############################################


alb = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=1)],
    # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1)],
                additional_targets = {'mask': 'mask'})

def alb_fn(image, mask):
    aug = alb(image=image, mask=mask)
    return aug['image'], aug['mask']


def augment(data, masks):
    print('Starting data augmentation')

    for i in range(data.shape[0]):
        data[i, ...], masks[i, ...] = alb_fn(data[i, ...], masks[i, ...])

        
##        # Random flip:
##        flip = np.random.choice([-1, 0, 1], 1).astype(int)
##        if flip >= 0:
##            data[i, ...] = np.flip(data[i, ...], axis=flip)
##            masks[i, ...] = np.flip(masks[i, ...], axis=flip)
##
##        # Random 90 deg rotation:
##        rot = np.random.choice([0, 1], 1).astype(int)
##        if rot > 0:
##            data[i, ...] = np.rot90(data[i, ...])
##            masks[i, ...] = np.rot90(masks[i, ...])

    print('Data augmentation complete')

    return data, masks
        

def norm_resnet(data):
    r = data[:, :, :, 0]
    g = data[:, :, :, 1]
    b = data[:, :, :, 2]

    r = (r - 123.68) / 58.393
    g = (g - 116.779) / 57.12
    b = (b - 103.939) / 57.375

    data = patch = np.stack([r, g, b], axis=3)
    return data
    

def from_subject(subject, slide_path, split):
    patches_direc = []
    masks_direc = []
    masks_control_direc = []

    patches_locs = []


    direcs_valid = []
    k = np.where((subjects.Subject == subject) & (subjects.Type == split))[0][0]
    print('k =', k)
    for i in range(4):
        shift_x = subjects[f'{direcs[i]}_x'][k]
        shift_y = subjects[f'{direcs[i]}_y'][k]
        if ~(np.isnan(shift_x) | np.isnan(shift_y)):
            direcs_valid.append(direcs[i])

    print(f'Subject {subject} valid directions: {direcs_valid}')

    for direc in direcs_valid:

        # k = np.where(subjects.Subject == subject)[0][0]
        shift_x = subjects[f'{direc}_x'][k]
        shift_y = subjects[f'{direc}_y'][k]
        if np.isnan(shift_x) |  np.isnan(shift_y):
            print(f'Subject {subject} - {direc} is not valid.')
            continue
        shift_x = int(shift_x)
        shift_y = int(shift_y)
        
        if os.path.exists(f'./{subject}/{direc} Secondary.jpg') & os.path.exists(f'./{subject}/{direc} Section.jpg'):
            path = f'./{subject}/{direc} Secondary.jpg'
        elif os.path.exists(f'./{subject}/{direc} Secondary Tissue.jpg') & os.path.exists(f'./{subject}/{direc} Section.jpg'):
            path = f'./{subject}/{direc} Secondary Tissue.jpg'
        else:
            print('Critical issue - must handle')
            continue
            
        sec = 255 - cv.imread(path)
        sil = 255 - cv.imread(f'./{subject}/{direc} Section.jpg')


        # Load whole slide data.
        slide = cv.imread(f'./{subject}/{slide_path}')

##        if slide.shape[0] > sec.shape[0]: # If slide is bigger than sec mask:
##            slide = cv.resize(slide, [sec.shape[1], sec.shape[0]])
##        else: # otherwise if the sec mask is bigger...
##            sec = cv.resize(sec, [slide.shape[1], slide.shape[0]])

        # Alignment of secondary mask:
        sec = np.roll(sec, shift_x, axis=0)
        sec = np.roll(sec, shift_y, axis=1)

        
        sec = np.sum(sec, axis=2)
        sec = (sec > 10)  # Convert to binary masks

        sil = np.sum(sil, axis=2)
        sil = sil > 10
        
        # Moving back to old resizing scheme...
        slide = cv.resize(slide, [sec.shape[1], sec.shape[0]])
        #slide = cv.cvtColor(slide, cv.COLOR_BGR2YUV)  # Converts to YUV color space. Better for ML...
        slide = cv.cvtColor(slide, cv.COLOR_BGR2RGB) # instead - convert to RGB for ResNet compatable.
        
##        # Closing holes in osteons
##        labeled_holes, n_holes = label(1 - sec)
##        sec[labeled_holes > 1] = 1

        ind = np.where(sil == 1)
        regx0 = np.min(ind[0])
        regx1 = np.max(ind[0])
        regy0 = np.min(ind[1])
        regy1 = np.max(ind[1])

        
        resx = np.floor((regx1 - regx0) / patch_size * increment).astype(int)
        resy = np.floor((regy1 - regy0) / patch_size * increment).astype(int)

        patches = []
        masks = []
        masks_control = []

        for i in range(resx):
            for j in range(resy):
                a = regx0 + i * patch_size // increment
                b = regy0 + j * patch_size // increment

                patch_loc = np.array([a, b])

                sil_sub = sil[a:a + patch_size, b:b + patch_size]


                if np.any(sil_sub == 0):
                    continue  # now - ONLY consider slides which are fully within. Othwrwise, will have false neg.

                sec_sub = sec[a:a + patch_size, b:b + patch_size]
                
                if sec_sub.shape != (patch_size, patch_size):
                    # print('Callout')
                    continue

                kernel = np.ones((3, 3), np.uint8)
                sil_sub = 1 - sil_sub
                sil_sub = sil_sub.astype(np.uint8)
                sil_sub = cv.dilate(sil_sub, kernel, iterations=1)


                # Now - de-blebbing the mask:
                # sec_sub = debleb(sec_sub)

                # sub, n_sec = label(sec_sub)
                # valid = np.zeros((patch_size, patch_size))
                valid = sec_sub

                
                masks.append(valid)

                h = slide[a:a + patch_size, b:b + patch_size]
                patches.append(h)


                patches_locs.append(patch_loc)
                

        patches = [get_patch(patches[i]) for i in range(len(patches))]
        masks = [get_mask(masks[i]) for i in range(len(masks))]

        patches = np.stack(patches, axis=0)
        masks = np.stack(masks, axis=0)

        patches_direc.append(patches)
        masks_direc.append(masks)
        
        # masks_control_direc.append(masks_control)
        # n_nzero = np.sum([1 for i in range(masks.shape[0]) if ~np.all(masks[i, :, :] == 0)]) 
        print(f'Completed: Subject {subject} - {direc}) | Patches found: {masks.shape[0]}')

    patches_direc = np.concatenate(patches_direc, axis=0)
    masks_direc = np.concatenate(masks_direc, axis=0)
    # masks_control_direc = np.concatenate(masks_control_direc, axis=0)

    # print('Number of patches:', patches_direc.shape[0])

    del sec, sil, slide, patches, masks  # Cleaning memory

    
    # Dropout zero, one slides, if interest is <5% or >95%
    # ind_zero = [i for i in range(masks_direc.shape[0]) if np.all(masks_direc[i, :, :] == 0)]
    # ind_ones = [i for i in range(masks_direc.shape[0]) if np.all(masks_direc[i, :, :] > 0)]

    ind_zero = [i for i in range(masks_direc.shape[0]) if np.sum(masks_direc[i, :, :]) < (0.05*n**2)]
    ind_ones = [i for i in range(masks_direc.shape[0]) if np.sum(masks_direc[i, :, :]) > (0.95*n**2)]
    ind_bad = np.concatenate([ind_zero, ind_ones]).astype(int)
    
    ind_new = np.arange(masks_direc.shape[0])
    ind_new = np.delete(ind_new, ind_bad)
    
    patches_final = patches_direc[ind_new, ...]
    masks_final = masks_direc[ind_new, ...]

    patches_locs = np.stack(patches_locs, axis=0)
    patches_locs = patches_locs[ind_new, ...]

    print('Patches excluded:', len(ind_zero) + len(ind_ones))
    print('Final number of patches:', len(ind_new))

    return patches_final, masks_final, patches_locs


print('Processing training dataset')
train_data = []
train_masks = []
subjects_train = np.array(subjects.Subject[subjects.Type == 'Train'])
paths_train = np.array(subjects.Path[subjects.Type == 'Train'])                       
for s in range(len(subjects_train)):
    data, masks, locs = from_subject(subjects_train[s], paths_train[s], split='Train')
    train_data.append(data)
    train_masks.append(masks)

##    os.chdir(base_dir)
##    np.save(f'./data/{subjects_train[s]}_data.npy', data)
##    np.save(f'./data/{subjects_train[s]}_masks.npy', masks)
##    os.chdir(data_dir)
    
        
train_data = np.concatenate(train_data, axis=0)
train_masks = np.concatenate(train_masks, axis=0)

# Used in testing: 
train_data, val_data, train_masks, val_masks = train_test_split(train_data, train_masks, test_size=0.1)


# Performing data augmentation
train_data, train_masks = augment(train_data, train_masks)

# Scrambling training data.
# print('Testy 1:', np.sum(train_data[0, :, :, :]))
ind_scramble = np.arange(train_data.shape[0])

np.random.shuffle(ind_scramble)
train_data = train_data[ind_scramble]
train_masks = train_masks[ind_scramble]
# print('Testy 2:', np.sum(train_data[0, :, :, :]))


##print('Processing validation dataset')
##val_data = []
##val_masks = []
##subjects_val = np.array(subjects.Subject[subjects.Type == 'Val'])
##paths_val = np.array(subjects.Path[subjects.Type == 'Val'])
##for s in range(len(subjects_val)):
##    data, masks, locs = from_subject(subjects_val[s], paths_val[s], split='Val')
##    val_data.append(data)
##    val_masks.append(masks)
##
####    os.chdir(base_dir)
####    np.save(f'./data/{subjects_test[s]}_data.npy', data)
####    np.save(f'./data/{subjects_test[s]}_masks.npy', masks)
####    os.chdir(data_dir)
##
##val_data = np.concatenate(val_data, axis=0)
##val_masks = np.concatenate(val_masks, axis=0)


print('Processing test dataset')
test_data = []
test_masks = []
test_locs = []
subjects_test = np.array(subjects.Subject[subjects.Type == 'Test'])
paths_test = np.array(subjects.Path[subjects.Type == 'Test'])
for s in range(len(subjects_test)):
    data, masks, locs = from_subject(subjects_test[s], paths_test[s], split='Test')
    test_data.append(data)
    test_masks.append(masks)
    test_locs = locs

##    os.chdir(base_dir)
##    np.save(f'./data/{subjects_test[s]}_data.npy', data)
##    np.save(f'./data/{subjects_test[s]}_masks.npy', masks)
##    os.chdir(data_dir)
        
    
test_data = np.concatenate(test_data, axis=0)
test_masks = np.concatenate(test_masks, axis=0)
test_locs = np.stack(test_locs, axis=0)


# Data normalization - for ResNet imput:
train_data = norm_resnet(train_data)
val_data = norm_resnet(val_data)
test_data = norm_resnet(test_data)


# Saving:
os.chdir(base_dir)

print('Saving patches')

np.save('./data/train_data.npy', train_data)
np.save('./data/train_masks.npy', train_masks)
np.save('./data/val_data.npy', val_data)
np.save('./data/val_masks.npy', val_masks)
np.save('./data/test_data.npy', test_data)
np.save('./data/test_masks.npy', test_masks)

np.save('./data/test_locs.npy', test_locs) # For post-hoc registration


print('Training dataset size:', train_data.shape)
print('Validation dataset size:', val_data.shape)
print('Testing dataset size:', test_data.shape)



