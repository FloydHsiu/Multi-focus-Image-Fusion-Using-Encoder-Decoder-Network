# Multi-focus Image Fusion Using Encoder-Decoder Network

## Description

This code is trained on TPU supported by TensorFlow Research Cloud (TFRC).
Thanks for TFRC that gave me the opportunity to access TPU.
By using TPU, I can train my model by setting a bigger mini-batch size to see if the result get better.
There are some introduction and results included in <https://link.medium.com/Iaqx4CmS83>

## Installation

### 1. Follow tutorial to set up your own **TPU** and **Compute Engine**

[Training ResNet on Cloud TPU (TF 2.x)](https://cloud.google.com/tpu/docs/tutorials/resnet-2.x)

You can get your **TPU_NAME** and **STORAGE_BUCKET**

### 2. Requirement

* Python 3.7
* pip install
  
```bash
tensorflow==2.0.1
tqdm
opencv-python  
```

### 3. Download dataset and unzip them

* Adobe Deep Matting Dataset: (Contact author of Deep Image Matting [6])
* COCO Dataset Train 2014: <http://cocodataset.org/#download>
* Lytro Multi-focus Dataset: <https://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset>

This code need three dataset. Adobe Deep Matting Dataset is a dataset with foreground images and precise foreground segmentations. COCO Dataset Train 2014 is a dataset we use it as background images when we compose multi-focus images. Lytro Multi-focus Dataset is a dataset that contain multi-focus images that were captured from real world.

1. Copy fg and alpha directories from

   `./Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Adobe-licensed`

   and

   `./Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/Other`

   to

   `./fg` and `./alpha`

2. Copy images from COCO Dataset Train 2014 to `./bg`

3. Run `python data_clean.py` to get rid of transparent objects in fg and alpha
  
### 4. Compose training data

```bash
python data.py \
--fg_dir='fg'\
--bg_dir='bg'\
--alpha_dir='alpha'\
--output='train.tfrecords'\
--num=100000
```

### 5. Upload training data

```bash
export DATA_PATH=$STORAGE_BUCKET/train.tfrecords
export LYTRO_DIR=$STORAGE_BUCKET/Inference
gsutil cp train.tfrecords $DATA_PATH
gsutil cp -r LytroDataset $LYTRO_DIR
```

## Usage

### Train your model

```bash
export $LOGS_DIR=$STORAGE_BUCKET/Logs
python train.py \
--tpu_name=$TPU_NAME\
--data_path=$DATA_PATH\
--logs_dir=$LOGS_DIR\
--lytro_dir=$LYTRO_DIR
```

### Observe training progress

```bash
tensorboard --logdir $LOGS_DIR/{YOUR_LOG_DIRECTORY}/tensorboard
```

## Reference

1. K. He , J. Sun and X. Tang , “Guided image filtering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 6, pp. 1397-1409, 2013
2. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C.L. Zitnick, "Microsoft COCO: common objects in context," European Conference on Computer Vision, pp. 740–755, 2014.
3. Y. Liu, X. Chen, H. Peng, and Z. Wang, "Multi-focus image fusion with a deep convolutional neural network," Information Fusion, vol. 36, pp. 191–207, 2017.
4. M. Nejati, S. Samavi, and S. Shirani, "Multi-focus image fusion using dictionary-based sparse representation," Information Fusion, vol. 25, pp. 72–84, 2015.
5. Odena, et al., "Deconvolution and checkerboard artifacts", Distill, 2016. <http://doi.org/10.23915/distill.00003>
6. N. Xu, B. Price, S. Cohen, and T. Huang, "Deep image matting," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
7. S. Zagoruyko and N. Komodakis, "Learning to compare image patches via convolutional neural networks," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
