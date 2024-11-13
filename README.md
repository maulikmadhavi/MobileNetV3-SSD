# MobileNetV3-SSD

MobileNetV3-SSD implementation in PyTorch

For the second version, please visit https://github.com/shaoshengsong/MobileNetV3-SSD-Compact-Version for test results.
For new technology, visit https://github.com/shaoshengsong/quarkdet for a lightweight object detection model.

**Purpose**
Object Detection

**Environment**

- OS: Ubuntu 18.04
- Python: 3.6
- PyTorch: 1.1.0

**Using MobileNetV3-SSD for Object Detection**

**Support Export ONNX**

Code references:

**1. SSD Part**

[A PyTorch Implementation of Single Shot MultiBox Detector](https://github.com/amdegroot/ssd.pytorch)

**2. MobileNetV3 Part**

[1. MobileNetV3 with PyTorch, provides pre-trained model](https://github.com/xiaolai-sqlai/mobilenetv3)

[2. MobileNetV3 in PyTorch and ImageNet pretrained models](https://github.com/kuan-wang/pytorch-mobilenet-v3)

[3. Implementing Searching for MobileNetV3 paper using PyTorch](https://github.com/leaderj1001/MobileNetV3-Pytorch)

[4. MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in PyTorch 1.0 / PyTorch 0.4. Out-of-box support for retraining on Open Images dataset. ONNX and Caffe2 support. Experiment Ideas like CoordConv. No discernible latency cost](https://github.com/qfgaohao/pytorch-ssd)

Note: Only MobileNetV3 is compatible here, not MobileNetV1 or MobileNetV2.

**Download Data**
This example uses Cake and Bread due to the small data size.
Total size of all categories is 561G, Cake and Bread is 3.2G.

```sh
python3 open_images_downloader.py --root /media/santiago/a/data/open_images --class_names "Cake,Bread" --num_workers 20
```

**Training Process**

**First Training**

```sh
python3 train_ssd.py --dataset_type open_images --datasets /media/santiago/data/open_images --net mb3-ssd-lite --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001 --batch_size 5
```

**Load Pre-trained Model**

```sh
python3 train_ssd.py --dataset_type open_images --datasets /media/santiago/data/open_images --net mb3-ssd-lite --pretrained_ssd models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 200 --base_net_lr 0.001 --batch_size 5
```

**Test an Image**

```sh
python run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth models/open-images-model-labels.txt /home/santiago/picture/test.jpg
```

**Video Detection**

```sh
python3 run_ssd_live_demo.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth models/open-images-model-labels.txt
```

**Cake and Bread Pretrained Model**

Link: https://pan.baidu.com/s/1byY1eJk3Hm3CTp-29KirxA 
Code: qxwv

**VOC Dataset Pretrained Model**

Link: https://pan.baidu.com/s/1yt_IRY0RcgSxB-YwywoHuA 
Code: 2sta
