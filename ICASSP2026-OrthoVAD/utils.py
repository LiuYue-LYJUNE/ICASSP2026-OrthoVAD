import numpy as np 
import os 
import matplotlib.pyplot as plt  
plt.switch_backend('agg')  
import torch

def random_extract(feat, t_max):
    r = np.random.randint(len(feat)-t_max)
    return feat[r:r+t_max], r

def random_extract_step(feat, t_max, step):
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r

def random_perturb(feat, length):
    samples = np.arange(length) * len(feat) / length
    for i in range(length):
        if i < length - 1:
            if int(samples[i]) != int(samples[i + 1]):
                samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
            else:
                samples[i] = int(samples[i])
        else:
            if int(samples[i]) < length - 1:
                samples[i] = np.random.choice(range(int(samples[i]), length))
            else:
                samples[i] = int(samples[i])
    return feat[samples.astype('int')], samples.astype('int')

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat

def process_feat(feat, length, step):
    if len(feat) > length:
        if step and step > 1:
            features, r = random_extract_step(feat, length, step)
            return pad(features, length), r
        else:
            features, r = random_extract(feat, length)
            return features, r
    else:
        return pad(feat, length), 0

def process_feat_sample(feat, length):
    if len(feat) > length:
            features, samples = random_perturb(feat, length)
            return features, samples
    else:
        return pad(feat, length), 0

def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False, width=15, height=5):
    if not os.path.exists(os.path.join(save_root, save_path, 'plot')):
        os.makedirs(os.path.join(save_root, save_path, 'plot'))
    for k, v in predict_dict.items():
        predict_np = v.repeat(16)
        k = k[:-2]
        label_np = label_dict[k][:len(predict_np)]
        x = np.arange(len(predict_np))
        plt.figure(figsize=(width, height))
        plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)
        label_np = np.array(label_np)
        plt.fill_between(x, label_np, where=label_np > 0, facecolor="r", alpha=0.3)

        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel('Frames')
        plt.ylabel('Anomaly scores')
        plt.grid(True, linestyle='-.')
        plt.legend()
        output_dir = os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, k + '.svg'), format='svg')
        plt.close()