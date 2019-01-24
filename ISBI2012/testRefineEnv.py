from refineEnv import *
from img_aug_func import *
import matplotlib.pyplot as plt
import cv2   
import skimage.io as io
import os, sys, argparse, glob
from natsort import natsorted
from skimage.morphology import erosion, binary_erosion
from skimage.filters import sobel

def get_cell_prob (lbl):
    elevation_map = []
    for img in lbl:
        elevation_map += [sobel (img)]
    elevation_map = np.array (elevation_map)
    elevation_map = elevation_map > 0
    cell_prob = (lbl > 0) ^ elevation_map
    for i in range (len (cell_prob)):
        cell_prob [i] = binary_erosion (cell_prob [i])
    return np.array (cell_prob, dtype=np.uint8) * 255

def get_data ():
    base_path = 'DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path) [0]
    y_train = read_im (train_label_path) [0]
    print (X_train.shape, y_train.shape)
    plt.show ()
    y_train = get_cell_prob (y_train)

    # for i in range (len (y_train)):
    for img_id in range (len (y_train)):
        y_train[img_id] = label (y_train[img_id] > 0)
    y_train =  y_train.astype (np.uint8)

    return X_train , y_train

def down_sample_3d (data_list, factor):
    ret = []
    for data in data_list:
        assert (len (data.shape) == 3)
        ret += [data[::2, ::2, ::2]]
    return ret

raw_list = []
lbl_list = []

def get_medical_env ():
    global raw_list, lbl_list
    if (len (raw_list) == 0):
        raw_list, lbl_list = get_data ()
        for i in range (len (lbl_list)):
            lbl_list[i] = label (lbl_list[i] > 0)
        lbl_list = lbl_list.astype (np.uint8)
    print ("DEBUG", len (np.unique (lbl_list)))
    SEG_checkpoints_paths = [
        'Unet/checkpoints_r/checkpoint_455260.pth.tar',
        'Unet/checkpoints_r/checkpoint_455260.pth.tar',
        'Unet/checkpoints_r/checkpoint_455260.pth.tar',
        'Unet/checkpoints_r/checkpoint_455260.pth.tar'
    ]

    DS_FACTORS = [0.4, 0.7, 1.0, 1.0]

    return Environment (raw_list, lbl_list, SEG_checkpoints_paths, DS_FACTORS)


player = get_medical_env ()

obs = player.reset ()


# obs, reward, done, info = player.step (2)
# obs, reward, done, info = player.step (3)
# obs, reward, done, info = player.step (1)
# obs, reward, done, info = player.step (3)
# obs, reward, done, info = player.step (1)

print ('obs shape:', obs.shape)
print ('render shape:', player.render ().shape)

done = False

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1000, 600)

stack = []

while not done:
    print ('level:', player.state.node.level, 'current score: ', player.cal_metric ())
    cv2.imshow ('image', player.render ())
    action = cv2.waitKey()

    action -= ord ('0')

    obs, reward, done, info = player.step (action)
    print ('action: ', action, 'reward', reward, 'done', done)
    if info['down_level']:
        stack += [info['current_score']]

    if info['up_level']:
        reward = player.cal_metric () - stack.pop ()
        print ('delayed reward', reward)
        done = False