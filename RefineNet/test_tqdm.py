from tqdm import tqdm
import time
import sys

n = int (1e7)

for i in range (3):
    pbar = tqdm (total=n, ascii=True)
    for j in range (n):
        pbar.update ()
    time.sleep (0.5)  
    pbar.close ()
    pbar.write (s = 'Finish a bar')
    
    
    
