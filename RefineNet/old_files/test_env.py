from environment import *
from img_aug_func import *
import matplotlib.pyplot as plt
import cv2   
import skimage.io as io

def get_data ():
    base_path = 'data/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    return X_train [0], y_train[0]

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
        raw_list, lbl_list = down_sample_3d ([raw_list, lbl_list], 2)
        lbl_list = label (erode_label ([lbl_list > 0]) [0])
        raw_list = np.pad (raw_list, 80, mode='constant', constant_values=0)
        lbl_list = np.pad (lbl_list, 80, mode='constant', constant_values=0)
        
    return Environment (raw_list, lbl_list)


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


while not done:
    cv2.imshow ('image', player.render ())
    action = cv2.waitKey()
    if (action == ord (' ')):
        obs = player.observation ()
        raw = obs[...,0]
        # xy, zx, zy
        raw = [raw, raw.transpose (1, 0, 2), raw.transpose (2, 0, 1)]
        raw = np.concatenate (raw, 2)
        loc = obs[...,2]
        loc = [loc, loc.transpose (1, 0, 2), loc.transpose (2, 0, 1)]
        loc = np.concatenate (loc, 2)

        obs = np.concatenate ([raw, loc], 1)

        io.imsave ('obs.tif', obs.astype (np.uint8))
        print ('saved observation')
        continue
    if (action == ord ('a')):
        action = 10
    elif action == ord ('s'):
        acition = 11
    else:
        action -= ord ('0')

    obs, reward, done, info = player.step (action)
    print ('action: ', action, 'reward', reward, 'done', done)