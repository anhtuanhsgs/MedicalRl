from . import *

def init_data ():
    base_path = '/home/Pearl/tuan/_Data/ml16-master/segmentation/data/'
    raw_path = base_path + 'train/images/'
    label_path = base_path + 'train/labels/'

    raw_files = natsorted (glob.glob (raw_path + '*.png'))
    lbl_files = natsorted (glob.glob (label_path + '*.png'))

    # print len (raw_files), len (lbl_files)

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

    return raw_list, lbl_list

def mri_heart (name):
    config = Config ()
    config.history_length = 1


    raw_list, lbl_list = init_data ()
    config.task_fn =  lambda : MedicalEnv (raw_list, lbl_list)
    config.eval_env = MedicalEnv (raw_list, lbl_list)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True) 

    config.network_fn = lambda: 

    
