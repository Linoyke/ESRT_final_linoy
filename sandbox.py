import os
import random
import numpy as np
import matplotlib.pyplot as plt

folder_path = "/Users/alonhelvits/pythonProject/ESRT/data/DIV2K_decoded/DIV2K_train_HR/DIV2K_train_LR_bicubic/X2"
npy_files = [file for file in os.listdir(folder_path) if file.endswith(".npy")]
random_files = random.sample(npy_files, 3)

for file in random_files:
    file_path = os.path.join(folder_path, file)
    data = np.load(file_path)
    #Display the images
    print(data.shape)

    plt.imshow(data)
    plt.show()
