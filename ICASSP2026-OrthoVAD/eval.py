import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from utils import anomap


def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):

    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = './result'

    if dataset == 'shanghaitech':
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)
        with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f:
            frame_label_dict = pickle.load(f)
        with open(file=os.path.join(label_dict_path, 'video_label_10crop.pickle'),
                mode='rb') as f:
            video_label_dict = pickle.load(f)
        all_predict_np = np.zeros(0)
        all_label_np = np.zeros(0)
        for k, v in predict_dict.items():
            base_video_name = k[:-2]
            if video_label_dict[k] == '[1.0]':
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            elif video_label_dict[k] == '[0.0]':
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score))
        if plot:
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5)
        if os.path.exists(os.path.join(save_root, save_path)) == 0:
            os.makedirs(os.path.join(save_root, save_path))
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score))

    if dataset == 'ucf-crime':
        label_dict_path = '{}/ucf-crime/GT'.format(args.dataset_path)
        with open(file=os.path.join(label_dict_path, 'ucf_gt_upgate.pickle'), mode='rb') as f:
            frame_label_dict = pickle.load(f)
        with open(file=os.path.join(label_dict_path, 'video_label_10crop.pickle'),
                mode='rb') as f:
            video_label_dict = pickle.load(f)
        all_predict_np = np.zeros(0)
        all_label_np = np.zeros(0)
        for k, v in predict_dict.items():
            base_video_name = k[:-2]
            if video_label_dict[k] == '[1.0]':
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            elif video_label_dict[k] == '[0.0]':
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score))
        if plot:
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5)
        if os.path.exists(os.path.join(save_root, save_path)) == 0:
            os.makedirs(os.path.join(save_root, save_path))
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score))

    if dataset == 'xd':
        label_dict_path = '{}/xd/GT'.format(args.dataset_path)

        with open(file=os.path.join(label_dict_path, 'frame_lable_dict.pickle'), mode='rb') as f:
            frame_label_dict = pickle.load(f)
        try:
            gt_path = os.path.join(args.dataset_path, 'xd/GT/xd_gt.npy')
            gt = np.load(gt_path)
        except FileNotFoundError:
            print(f"Error:Can't find xd_gt.npy in {gt_path}")
            exit()

        all_predict_list = []
        all_label_list = []
        abnormal_predict_list = []
        abnormal_label_list = []

        current_gt_index = 0

        print("Dynamically reconstructing frame labels based on predictions....")
        for k, v in predict_dict.items():
            video_predictions = v.repeat(16)
            num_frames = len(video_predictions)
            if current_gt_index + num_frames > len(gt):
                print(f"Warning: The number of predicted frames ({num_frames}) + the current index ({current_gt_index}) exceeds the total length of the Ground Truth ({len(gt)}).")
                print(f"The labels for video {k} will be truncated. Please check the order of your predict_dict or verify if your gt file is correct.")
                num_frames = len(gt) - current_gt_index
                video_predictions = video_predictions[:num_frames]

            video_labels = gt[current_gt_index: current_gt_index + num_frames]

            current_gt_index += num_frames

            all_predict_list.append(video_predictions)
            all_label_list.append(video_labels)

            if 'label_A' not in k:
                abnormal_predict_list.append(video_predictions)
                abnormal_label_list.append(video_labels)

        all_predict_np = np.concatenate(all_predict_list)
        all_label_np = np.concatenate(all_label_list).astype(float)

        if len(np.unique(all_label_np)) > 1:
            all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
            pre, rec, _ = precision_recall_curve(all_label_np, all_predict_np)
            pr_auc = auc(rec, pre)
            print(f'Iteration: {itr} -> Global AP: {pr_auc:.4f}')
            print(f'Iteration: {itr} -> Global AUC: {all_auc_score:.4f}')
        else:
            all_auc_score, pr_auc = "N/A", "N/A"
            print("Cannot calculate global AUC/AP because all labels belong to the same class.")

        if abnormal_predict_list:
            abnormal_predict_np = np.concatenate(abnormal_predict_list)
            abnormal_label_np = np.concatenate(abnormal_label_list).astype(float)
            if len(np.unique(abnormal_label_np)) > 1:
                abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
                print(f'Iteration: {itr} -> Abnormal Video AUC (AUC_A): {abnormal_auc_score:.4f}')

                pre_a, rec_a, _ = precision_recall_curve(abnormal_label_np, abnormal_predict_np)
                pr_auc_abnormal = auc(rec_a, pre_a)
                print(f'Iteration: {itr} -> Abnormal Video AP (AP_A): {pr_auc_abnormal:.4f}')
            else:
                abnormal_auc_score = "N/A"
                pr_auc_abnormal = "N/A"
                print("Unable to calculate anomalous video metrics because all labels within the anomalous videos belong to the same category (likely all normal frames)")
        else:
            abnormal_auc_score = "N/A"
            pr_auc_abnormal = "N/A"
            print("No anomalous videos found, unable to calculate anomalous video metrics.")

        if not os.path.exists(os.path.join(save_root, save_path)):
            os.makedirs(os.path.join(save_root, save_path))

        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
            f.write(f'Iteration_{itr}_AUC: {all_auc_score}\n')
            f.write(f'Iteration_{itr}_AP: {pr_auc}\n')
            f.write(f'Iteration_{itr}_AUC_Abnormal: {abnormal_auc_score}\n')
            f.write(f'Iteration_{itr}_AP_Abnormal: {pr_auc_abnormal}\n')