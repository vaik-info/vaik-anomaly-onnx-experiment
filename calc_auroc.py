import argparse
import os
import glob
import json
from sklearn import metrics
import numpy as np

def calc_auroc(json_dict_list):
    gt_labels = [json_dict['answer'] == 'anomaly' for json_dict in json_dict_list]
    mse_mean_list = [json_dict['score'] for json_dict in json_dict_list]
    sort_indexes = np.argsort(mse_mean_list).tolist()
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_labels, mse_mean_list
    )
    auroc_metric = metrics.roc_auc_score(
        gt_labels, mse_mean_list
    )

    sort_json_dict_list = [json_dict_list[sort_index] for sort_index in sort_indexes]
    return auroc_metric, fpr, tpr, thresholds, sort_json_dict_list


def main(input_json_dir_path):
    json_path_list = glob.glob(os.path.join(input_json_dir_path, '*.json'))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)

    auroc_metric, fpr, tpr, thresholds, sort_json_dict_list = calc_auroc(json_dict_list)

    print(f'fpr: {fpr}')
    print(f'tpr: {tpr}')
    print(f'thresholds: {thresholds}')
    print(f'sort_score:')
    for sort_json_dict in sort_json_dict_list:
        print(f'{sort_json_dict["answer"]}: {sort_json_dict["score"]}: {sort_json_dict["image_path"]}')
    print(f'auroc_metric: {auroc_metric}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid_inference')
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)

    main(**args.__dict__)