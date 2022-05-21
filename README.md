# 貓狗辨識(using ResNet50)
## 使用環境以及語言
- #### Environment
> Anaconda
- #### Programming Language
> Python 
> Tensorflow 
> Opencv-contrib-python 

## 程式說明
使用dataset(ASIRRA)的圖片作為訓練根據，training set 和 validation set的比例為8:2。
會使用data augmentation增加正確率。

## 程式功能
![](https://i.imgur.com/DnonQWL.png)

- #### Show Model Structure(圖片為部分結構)
> ![](https://i.imgur.com/utYiTPx.png)

- #### Show TensorBoard
> ![](https://i.imgur.com/HcPieSV.png)

- #### Test
> 取出測試集的其中一張照片判斷是貓還是狗
> ![](https://i.imgur.com/HCKgdDd.png)


