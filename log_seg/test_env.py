from refineEnv import *
from img_aug_func import *
import matplotlib.pyplot as plt
import cv2   
import skimage.io as io
import os
from model import DQN

SEG_CHECKPOINT_PATH = 'checkpoints/checkpoint_250290.pth.tar'
BORDER_SIZE = 80
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def get_data ():
    base_path = 'DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    X_train = X_train [0] / 65535.0 * 255
    y_train = y_train [0]

    return X_train, y_train

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
        lbl_list = label (erode_label ([lbl_list > 0]) [0])
      
    return Environment (raw_list, lbl_list, SEG_CHECKPOINT_PATH)


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

def save_obs_tif (obs):
    raw = obs[...,0]
    # xy, zx, zy
    raw = [raw, raw.transpose (1, 0, 2), raw.transpose (2, 0, 1)]
    raw = np.concatenate (raw, 2)
    loc = obs[...,3]
    loc = [loc, loc.transpose (1, 0, 2), loc.transpose (2, 0, 1)]
    loc = np.concatenate (loc, 2)
    mask = obs [..., 1]
    mask = [mask, mask.transpose (1, 0, 2), mask.transpose (2, 0, 1)]
    mask = np.concatenate (mask, 2)

    obs = np.concatenate ([raw, mask, loc], 1)

    io.imsave ('obs.tif', obs.astype (np.uint8))
    print ('saved observation')

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dqn_device = torch.device("cuda:0")
q = DQN (4, 11, player.observation_space.shape).type(dtype).to(dqn_device)
q.eval ()

while not done:
    cv2.imshow ('image', player.render ())
    action = cv2.waitKey()
    # Save observation volume
    if (action == ord (' ')):
        obs = player.observation ()
        save_obs_tif (obs)
        continue
    if (action == ord ('a')): # Refine
        action = 9
    elif action == ord ('s'): # Back
        action = 10
    elif action == ord ('q'):
        player.debug ()
        continue
    else:
        action -= ord ('0') # Zoomin


    obs, reward, done, info = player.step (action)
    with torch.no_grad ():
        obs_t = obs.transpose (3, 0, 1, 2)
        obs_t = torch.tensor (obs_t[None], dtype=torch.float32, device=dqn_device) / 255.0
        print (q (obs_t))
    print ('action: ', action, 'reward', reward, 'done', done)

    if info['up_level']:
        done = False



    # if info['down_level']:
    #     stack += [info['current_score']]

    # if info['up_level']:
    #     reward =  stack.pop ()
    #     print ('delayed reward', reward)
    #     done = False