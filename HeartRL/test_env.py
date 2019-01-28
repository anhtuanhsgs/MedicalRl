from onestep_environment import *
from img_aug_func import *
import matplotlib.pyplot as plt


base_path = '/home/Pearl/tuan/_Data/ml16-master/segmentation/data/'
raw_path = base_path + 'test/images/'
label_path = base_path + 'test/labels/'

raw_files = natsorted (glob.glob (raw_path + '*.png'))
lbl_files = natsorted (glob.glob (label_path + '*.png'))

raw_list = read_im (raw_files)
lbl_list = read_im (lbl_files)
i = 0

while i < len (raw_list):
	if np.sum (lbl_list[i]) == 0:
		del lbl_list [i]
		del raw_list [i]
	else:
		i += 1

for i in range (len (raw_list)):
	raw_list[i] = np.squeeze (raw_list[i][:,:,0])

player = Environment (raw_list, lbl_list)

obs = player.reset ()

def show (obs):
    tmp = []
    for i in range (2):
        tmp += [obs[...,i]]
    img= np.concatenate (tmp, 1)
    plt.imshow (img, cmap='gray')
    plt.show ()

T = 3
for t in range (T):
	done = False
	while not Done:
		action = int (input ("action = "))
		obs, reward, done, info = player.step ()
		print ("reward: ", reward)
		show (obs)
