# ECG_Classification
心电时序多标签分类

## 数据预处理

- 原始数据

心电序列  data/source_data/hf_round2_train/100001.txt
```
I II V1 V2 V3 V4 V5 V6
12 27 14 54 11 6 20 4
13 27 14 54 11 6 20 4
14 28 14 55 12 7 20 4
15 29 14 55 12 8 20 4
16 30 14 56 13 9 20 4
17 30 14 56 13 9 20 4
19 31 15 57 14 10 21 5
20 31 14 57 14 10 21 5
21 32 14 57 14 10 21 5
22 32 13 57 14 10 21 5
23 32 13 57 14 10 21 5
24 33 12 58 15 11 21 6
24 33 11 58 15 11 21 7
24 33 11 58 15 11 21 7
24 33 10 58 15 11 21 8
24 33 10 58 15 11 21 8
...
```
![image](https://github.com/JagerLee/ECG-Classification/blob/main/data/0.png)
![image](https://github.com/JagerLee/ECG-Classification/blob/main/data/1.png)

异常事件标签  data/source_data/hf_round2_arrythmia.txt
```
QRS低电压
电轴右偏
起搏心律
T波改变
电轴左偏
心房颤动
非特异性ST段异常
下壁异常Q波
前间壁R波递增不良
ST段改变
一度房室传导阻滞
左束支传导阻滞
右束支传导阻滞
完全性左束支传导阻滞
左前分支传导阻滞
右心房扩大
短PR间期
左心室高电压
窦性心动过缓
早期复极化
窦性心律
融合波
ST-T改变
非特异性ST段与T波异常
快心室率
非特异性T波异常
室性早搏
房性早搏
窦性心律不齐
完全性右束支传导阻滞
窦性心动过速
不完全性右束支传导阻滞
顺钟向转位
逆钟向转位
```

- 标签统计筛选
```
>>> python label_select.py -h

usage: label_select.py [-h] [-th THRESHOLD]

select label above threshold.

optional arguments:
  -h, --help            show this help message and exit
  -th THRESHOLD, --threshold THRESHOLD
                        Threshold of label select. Default: 0.99
```

- 数据处理
```
>>> python data_preprocess.py -h

usage: data_preprocess.py [-h] [-t TEST_SIZE]

data preprocess of ecg.

optional arguments:
  -h, --help            show this help message and exit
  -t TEST_SIZE, --test TEST_SIZE
                        Test size of data set split. Default: 0.2
```

## 训练

```
>>> python train.py -h

usage: train.py [-h] [-b BATCH_SIZE] [-lr LR] [-e EPOCH]

train ResNet from ecg data.

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch BATCH_SIZE
                        Batch size of training. Default: 8
  -lr LR, --lr LR       Learning rate of training. Default: 0.01
  -e EPOCH, --epoch EPOCH
                        Max epoch of training. Default: 10
```

## 验证

```
>>> python evaluate.py
```

## 预测
```
>>> python predict.py -h

usage: predict.py [-h] [-m MODEL_PATH] [-i INPUT_PATH]

predict ecg data to label.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model MODEL_PATH
                        Model path of predict.
  -i INPUT_PATH, --input INPUT_PATH
                        Input file or dictionary path of predict.
```
<<<<<<< HEAD
=======
<<<<<<< HEAD

=======
>>>>>>> ecd69b11fa612ae29ffde57042e25afe8b8cf078
>>>>>>> dcdbc0b (update)
