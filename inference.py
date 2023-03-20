import argparse
import os
import glob
import json
import tqdm
import numpy as np
from PIL import Image
from tqdm import tqdm
from vaik_anomaly_onnx_inference.onnx_model import OnnxModel

def read_image_dir(image_dir_path):
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(image_dir_path, '**', files), recursive=True))
    image_path_list = sorted(image_path_list)
    image_list = []
    for image_path in tqdm(image_path_list, 'read images'):
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image_list.append(image)
    return image_list, image_path_list

def main(input_saved_model_path, batch_size, test_good_image_dir_path, test_anomaly_image_dir_path, output_json_dir_path):
    os.makedirs(output_json_dir_path, exist_ok=True)

    model = OnnxModel(input_saved_model_path)

    test_good_image_list, test_good_image_path_list = read_image_dir(test_good_image_dir_path)
    test_anomaly_image_list, test_anomaly_image_path_list = read_image_dir(test_anomaly_image_dir_path)

    import time
    start = time.time()
    test_good_output, test_good_raw_pred = model.inference(test_good_image_list, batch_size=batch_size)
    test_anomaly_output, test_anomaly_raw_pred = model.inference(test_anomaly_image_list, batch_size=batch_size)
    end = time.time()
    print(f'{(len(test_good_image_list)+len(test_anomaly_image_list))/(end-start)}[images/sec]')

    for image_path, output_elem in zip(test_good_image_path_list, test_good_output):
        sub_dir_name = image_path.split('/')[-2]
        output_json_path = os.path.join(output_json_dir_path, f'{sub_dir_name}_{os.path.splitext(os.path.basename(image_path))[0]}'+'.json')
        output_elem['answer'] = 'good'
        output_elem['image_path'] = image_path
        with open(output_json_path, 'w') as f:
            json.dump(output_elem, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    for image_path, output_elem in zip(test_anomaly_image_path_list, test_anomaly_output):
        sub_dir_name = image_path.split('/')[-2]
        output_json_path = os.path.join(output_json_dir_path, f'{sub_dir_name}_{os.path.splitext(os.path.basename(image_path))[0]}'+'.json')
        output_elem['answer'] = 'anomaly'
        output_elem['image_path'] = image_path
        with open(output_json_path, 'w') as f:
            json.dump(output_elem, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_path', type=str, default='~/.vaik_anomaly_pb_trainer/output/model.onnx')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_good_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid/good')
    parser.add_argument('--test_anomaly_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid/anomaly')
    parser.add_argument('--output_json_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid_inference')
    args = parser.parse_args()

    args.input_saved_model_path = os.path.expanduser(args.input_saved_model_path)
    args.test_good_image_dir_path = os.path.expanduser(args.test_good_image_dir_path)
    args.test_anomaly_image_dir_path = os.path.expanduser(args.test_anomaly_image_dir_path)
    args.output_json_dir_path = os.path.expanduser(args.output_json_dir_path)

    main(**args.__dict__)