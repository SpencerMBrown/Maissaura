import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.model_selection import train_test_split

from scipy.ndimage import label
from skimage.morphology import convex_hull_image


def add_buffer(input):
    output = np.zeros((n, n))
    output[buffer:(patch_size + buffer), buffer:(patch_size + buffer)] = input
    return output


def get_mask(mask):
    mask = add_buffer(mask)
    return mask


def get_patch(patch):
    y, u, v = cv.split(patch)

    y = cv.GaussianBlur(y, (5, 5), 0)
    u = cv.GaussianBlur(u, (5, 5), 0)
    v = cv.GaussianBlur(v, (5, 5), 0)

    y = cv.equalizeHist(y)
    u = cv.equalizeHist(u)
    v = cv.equalizeHist(v)

    canny = cv.Canny(y, 50, 100)

    sobel_x = cv.Sobel(y, cv.CV_64F, 1, 0, ksize=3)
    sobel_x = cv.convertScaleAbs(sobel_x)
    sobel_y = cv.Sobel(y, cv.CV_64F, 0, 1, ksize=3)
    sobel_y = cv.convertScaleAbs(sobel_y)

    # Adding buffers:
    y = add_buffer(y)
    u = add_buffer(u)
    v = add_buffer(v)
    canny = add_buffer(canny)
    sobel_x = add_buffer(sobel_x)
    sobel_y = add_buffer(sobel_y)

    patch = np.stack([y, u, v, canny, sobel_x, sobel_y], axis=2)
    return patch


# path = r'D:\Check Lab\Projects\OSU-CHS Projects\2018-05 Maiasaura Humeri\New Project Code Images'
# os.chdir(path)

#Global variables for run
# subject = 'H40'

patch_size = 64
buffer = 16 # on either side.
n = patch_size + 2*buffer # input to ML.

batch_size = 5
epochs = 10

direcs = ['North', 'South', 'East', 'West']


def from_subject(subject):
    patches_direc = []
    masks_direc = []
    masks_control_direc = []

    for direc in direcs:
        path = f'./{subject}/{direc} Secondary.jpg'
        if not os.path.exists(path):
            print(f'Subject {subject} - {direc} is not found.')
            continue

        sec = 255 - cv.imread(f'./{subject}/{direc} Secondary.jpg')
        sil = 255 - cv.imread(f'./{subject}/{direc} Section.jpg')

        sec = np.sum(sec, axis=2)
        sec = (sec > 10)  # Convert to binary mask.

        sil = sil[:, :, 0]
        sil = sil > 10

        # Load whole slide data.
        slide = cv.imread(f'./{subject}/{subject}-1 Full 2x Rescan resize.tif')
        slide = cv.resize(slide, [sec.shape[1], sec.shape[0]])
        slide = cv.cvtColor(slide, cv.COLOR_BGR2YUV)  # Converts to YUV color space. Better for ML...

        # Closing holes in osteons
        labeled_holes, n_holes = label(1 - sec)
        sec[labeled_holes > 1] = 1

        ind = np.where(sil == 1)
        regx0 = np.min(ind[0])
        regx1 = np.max(ind[0])
        regy0 = np.min(ind[1])
        regy1 = np.max(ind[1])

        increment = 4  # For data augmentation - 8x data increment.
        resx = np.floor((regx1 - regx0) / patch_size * increment).astype(int)
        resy = np.floor((regy1 - regy0) / patch_size * increment).astype(int)

        patches = []
        masks = []
        masks_control = []

        for i in range(resx):
            for j in range(resy):
                a = regx0 + i * patch_size // increment
                b = regy0 + j * patch_size // increment

                sil_sub = sil[a:a + patch_size, b:b + patch_size]

                # if (np.sum(sil_sub) / (patch_size**2)) < (1 - 1/increment):
                #     continue
                if np.any(sil_sub == 0):
                    continue  # now - ONLY consider slides which are fully within. Othwrwise, will have false neg.

                sec_sub = sec[a:a + patch_size, b:b + patch_size]
                # if np.sum(sec_sub) == 0:
                #     continue

                kernel = np.ones((3, 3), np.uint8)
                sil_sub = 1 - sil_sub
                sil_sub = sil_sub.astype(np.uint8)
                sil_sub = cv.dilate(sil_sub, kernel, iterations=1)

                sub, n_sec = label(sec_sub)
                valid = np.zeros((patch_size, patch_size))

                edge_mask = np.ones((patch_size, patch_size))
                edge_mask[1:-1, 1:-1] = 0
                edge_mask = edge_mask.astype(np.uint8)

                for k in range(n_sec):
                    segment = sub == k  # mask for specific shape.
                    # intersects_sil = segment & sil_sub # Intersection with silouette.
                    # intersects_sil = np.sum(intersects_sil)
                    intersects_edge = segment & edge_mask
                    intersects_edge = np.sum(intersects_edge)

                    if np.sum(segment) < 10:
                        continue  # Filter out segments that are too small - not actually osteons, just noise.
                    # if intersects_sil > 0:
                    #     continue # Don't use segment if touching the silouette edge
                    if intersects_edge > 0:
                        continue  # Don't use segment if touching patch edge.

                    # Now - check if there is >1 concavity defects.
                    hull = convex_hull_image(segment)
                    defects = (1 - segment) * hull

                    defects, n_defects_raw = label(defects)

                    n_defects = 0
                    for w in range(1, n_defects_raw):
                        defect_size = np.sum(defects == w)

                        if defect_size >= 10:  # cutoff for defect registartion is 10 pixels...
                            n_defects += 1

                    if n_defects > 1:
                        continue  # don't use segment if more than 1 concavity defect

                    valid = valid + segment  # adds a valid segment to the final mask.

                masks.append(valid)
                masks_control.append(add_buffer(sec_sub))

                h = slide[a:a + patch_size, b:b + patch_size]
                patches.append(h)

        patches = [get_patch(patches[i]) for i in range(len(patches))]
        masks = [get_mask(masks[i]) for i in range(len(masks))]

        patches = np.stack(patches, axis=0)
        masks = np.stack(masks, axis=0)

        patches_direc.append(patches)
        masks_direc.append(masks)
        # masks_control_direc.append(masks_control)

        print(f'Completed: Subject {subject} - {direc}, Patches found: {len(patches)}')

    patches_direc = np.concatenate(patches_direc, axis=0)
    masks_direc = np.concatenate(masks_direc, axis=0)
    # masks_control_direc = np.concatenate(masks_control_direc, axis=0)

    print('Number of patches:', patches_direc.shape[0])

    del sec, sil, slide, patches, masks  # Cleaning memory


    # Randomly sampling out most of the zero masks - so non-zero = 0.

    ind_zero = [i for i in range(masks_direc.shape[0]) if np.all(masks_direc[i, :, :]==0)]
    ind_nonzero = np.array([i for i in range(masks_direc.shape[0]) if ~np.all(masks_direc[i, :, :]==0)])

    n_nonzero = len(ind_nonzero)
    ind_zero = np.random.choice(ind_zero, size=n_nonzero, replace=False)

    ind_new = np.concatenate([ind_nonzero, ind_zero])
    np.random.shuffle(ind_new)

    patches_final = patches_direc[ind_new, :, :, :]
    masks_final = masks_direc[ind_new, :, :]
    # masks_control_final = masks_control_direc[ind_new, :, :]

    return patches_final, masks_final


# TODO: FIX THIS, sort through directories.
subjects = ['H40', 'H40', 'H40']
subjects_train, subjects_test = train_test_split(subjects, test_size=0.3)
subjects_train, subjects_val = train_test_split(subjects_train, test_size=0.3)

train_data = []
train_masks = []

for s in subjects_train:
    train_data_sub, train_masks_sub = from_subject(s)
    train_data.append(train_data_sub)
    train_masks.append(train_masks_sub)

train_data = np.stack(train_data, axis=0)
train_masks = np.stack(train_masks, axis=0)

test_data = []
test_masks = []

for s in subjects_test:
    test_data_sub, test_masks_sub = from_subject(s)
    test_data.append(test_data_sub)
    test_masks.append(test_masks_sub)

test_data = np.stack(test_data, axis=0)
test_masks = np.stack(test_masks, axis=0)

val_data = []
val_masks = []

for s in subjects_val:
    val_data_sub, val_masks_sub = from_subject(s)
    val_data.append(val_data_sub)
    val_masks.append(val_masks_sub)

val_data = np.stack(val_data, axis=0)
val_masks = np.stack(val_masks, axis=0)


# Filtering reductino:
# print('Control:', np.sum(masks_control_final) / np.prod(masks_control_final.shape))
# print('Filtered:', np.sum(masks_final) / np.prod(masks_final.shape))


# Split data:
# train_data, test_data, train_masks, test_masks = train_test_split(patches_final, masks_final, test_size=0.1)
# train_data, val_data, train_masks, val_masks = train_test_split(train_data, train_masks, test_size=0.1)

# Saving:
# os.chdir(r'C:\Users\researchlabs\Documents\spencers_stuff')

np.save('./models/train_data.npy', train_data)
np.save('./models/train_masks.npy', train_masks)
np.save('./models/test_data.npy', test_data)
np.save('./models/test_masks.npy', test_masks)
np.save('./models/val_data.npy', val_data)
np.save('./models/val_masks.npy', val_masks)
# Note - should create train pos rate from train_masks, NOT the comp.
# np.save('./models/masks.npy', masks_final)

print('Training subjects:')
print(subjects_train)
print('Testing masks:')
print(subjects_test)
print('Validation masks:')
print(subjects_val)