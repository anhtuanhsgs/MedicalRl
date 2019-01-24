from environment import *
from Utils.img_aug_func import *
from natsort import natsorted
from skimage import io
from skimage.measure import label
import matplotlib.pyplot as plt


def spliter (raw, prob):
    ret = (prob > (255 * 0.85)).astype (np.float32) * 255.0
    return ret

def merger (raw, prob):
    ret = (prob > (255 * 0.55)).astype (np.float32) * 255.0
    return ret

env_config = {
    'corrector_size': [128, 128], 
    'spliter': spliter,
    'merger': merger,
    'cell_thres': int (255 * 0.7),
    'T': 100,
    'agent_out_shape': [2, 16, 16],
    'num_feature': 3,
}


def get_data (path, args):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train


raw , gt_lbl = get_data (path='Data/train/', args=None)
prob = io.imread ('Data/train-membranes-idsia.tif')

raw = raw [:10]
gt_lbl = gt_lbl [:10]
prob = prob [:10]

lbl = []
for img in prob:
    lbl += [label (img > env_config ['cell_thres'])]
lbl = np.array (lbl)
print (lbl.shape)
env = EM_env (raw, lbl, prob, env_config, 'train', gt_lbl)

done = False
obs = env.reset ()

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()

sum_score = 0
for y in range (16):
    for x in range (16):
        a = 0

        shape = env_config ['agent_out_shape']
        action = (shape[1] * shape[2]) * a + shape[2] * y + x
        
        obs, reward, done, info = env.step (action)
        sum_score += reward
        print ('action: ', action) 
        print ('reward: ', reward)

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()

