import numpy as np

save_dir = "/home/fate/gitproject/Output_Prediction/data/exp2_train_grain128a_round256" 
plains = np.load(f"{save_dir}/plain_texts.npy", allow_pickle=False)
labels = np.load(f"{save_dir}/cipher_texts.npy", allow_pickle=False)

print("plains:", plains.shape, plains.dtype) 
print("labels:", labels.shape, labels.dtype)
print("plains[1]:", plains[1]) 
print("labels[1]:", labels[1] if labels.ndim > 1 else labels[10:20])
