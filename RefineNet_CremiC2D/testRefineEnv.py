from refineEnv import *
from img_aug_func import *
import matplotlib.pyplot as plt
import cv2   
import skimage.io as io

def get_data ():
    base_path = 'DATA/'
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
        for i in range (len (lbl_list)):
            lbl_list[i] = label (lbl_list[i] > 0)
        lbl_list = lbl_list.astype (np.uint8)
    print ("DEBUG", len (np.unique (lbl_list)))
    SEG_checkpoints_paths = [
        'FCN/checkpoints/128_4/checkpoint_1113000.pth.tar',
        'FCN/checkpoints/128_2/checkpoint_550250.pth.tar',
        'FCN/checkpoints/128_1/checkpoint_496500.pth.tar'
    ]

    return Environment (raw_list, lbl_list, SEG_checkpoints_paths)


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