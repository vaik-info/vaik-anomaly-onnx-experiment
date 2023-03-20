# vaik-anomaly-onnx-experiment

Create json file by anomaly model. Calc ACC.


## Install

```shell
pip install -r requirements.txt
```

## Usage

### Create json file

```shell
python inference.py --input_saved_model_path ~/.vaik_anomaly_pb_trainer/output/model.onnx \
                --batch_size 8 \
                --test_good_image_dir_path ~/.vaik-mnist-anomaly-dataset/valid/good \
                --test_anomaly_image_dir_path ~/.vaik-mnist-anomaly-dataset/valid/anomaly \
                --output_json_dir_path ~/.vaik-mnist-anomaly-dataset/valid_inference
```

- test_good_image_dir_path & test_anomaly_image_dir_path
    - example

```shell
~/.vaik-mnist-anomaly-dataset$ tree
.
└── valid
    ├── anomaly
    │   ├── 00000000.png
    │   ├── 00000001.png
    │   ├── 00000002.png
    │   ├── 00000003.png
    │   ├── 00000004.png
    │   ├── 00000005.png
    │   ├── 00000006.png
    │   ├── 00000007.png
    │   ├── 00000008.png
    │   └── 00000009.png
    └── good
        ├── 00000000.png
        ├── 00000001.png
        ├── 00000002.png
        ├── 00000003.png
        ├── 00000004.png
        ├── 00000005.png
        ├── 00000006.png
        ├── 00000007.png
        ├── 00000008.png
        └── 00000009.png
```

#### Output
- output_json_dir_path
    - example

```json
{
  "answer": "anomaly",
  "image_path": "~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000002.png",
  "score": 0.044695959791126394
}
```
-----

### Calc AUROC

```shell
python calc_auroc.py --input_json_dir_path '~/.vaik-mnist-anomaly-dataset/valid_inference'
```

#### Output

``` text
fpr: [0.  0.  0.  0.1 0.1 1. ]
tpr: [0.  0.1 0.9 0.9 1.  1. ]
thresholds: [1.0565346  0.0565346  0.02602923 0.02546406 0.02277262 0.01077769]
sort_score:
good: 0.010777691303616575: ~/.vaik-mnist-anomaly-dataset/valid/good/00000006.png
good: 0.01148976980536613: ~/.vaik-mnist-anomaly-dataset/valid/good/00000002.png
good: 0.012157781460961102: ~/.vaik-mnist-anomaly-dataset/valid/good/00000005.png
good: 0.012827456449687085: ~/.vaik-mnist-anomaly-dataset/valid/good/00000001.png
good: 0.01532052240899165: ~/.vaik-mnist-anomaly-dataset/valid/good/00000009.png
good: 0.01699300347659618: ~/.vaik-mnist-anomaly-dataset/valid/good/00000003.png
good: 0.019713465396788078: ~/.vaik-mnist-anomaly-dataset/valid/good/00000007.png
good: 0.02251594867192428: ~/.vaik-mnist-anomaly-dataset/valid/good/00000004.png
good: 0.022749472375797372: ~/.vaik-mnist-anomaly-dataset/valid/good/00000000.png
anomaly: 0.022772624663008032: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000005.png
good: 0.025464061128976572: ~/.vaik-mnist-anomaly-dataset/valid/good/00000008.png
anomaly: 0.02602922915298551: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000004.png
anomaly: 0.02801276928466363: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000003.png
anomaly: 0.031817708899292495: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000000.png
anomaly: 0.03293825461838599: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000007.png
anomaly: 0.041977014921913136: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000006.png
anomaly: 0.043274505510364034: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000008.png
anomaly: 0.044695959791126394: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000002.png
anomaly: 0.049917189505395816: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000001.png
anomaly: 0.056534600978633165: ~/.vaik-mnist-anomaly-dataset/valid/anomaly/00000009.png
auroc_metric: 0.99
```