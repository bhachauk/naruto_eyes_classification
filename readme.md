### Various Neural Networks for Image Classification
---

##### Getting Started :
---

Classification on Naruto character eye images with various **CNN** for analysing the Neural Network accuracy.

**Classes :**
---

- Sharingan
- Byakugan
- Sage
- Others

**Networks :**
---

- CNN1
- CNN3
- CNN4
- LeNet
- VGG16


##### How to run ?
---

1. Train :

```python
python train.py -d data/train/
```

2. Test :

```python
python test.py -m modules/LENET_naruto_eye.h5 -i data/test/sage_draw.jpeg
```

##### Result :
---

![result](/data/naruto_eyes_prediction.gif)



##### References :
---

- https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
- https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
- https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
- https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
- https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/