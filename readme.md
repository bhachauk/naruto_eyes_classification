## Naruto Eyes Classifier using Neural Networks
---

<p style="align: center">
<img src="/data/naruto_eyes_prediction-web.gif" width="50%"/>
</p>

### Getting Started :
---

Classification on Naruto character eye images with various **CNN** for analysing the Neural Network accuracy.

###### Classes :
---

- Sharingan
- Byakugan
- Others (Sage)


### How to Train model with custom data?
---


1. Prepare Images in directory structure

```commandline
- train
    |_sharingan
    |_byakugan
    |_Others

```

2. Train Command:

```python
python train.py -d data/train/
```

3. Test :

```python
python test.py -m modules/LENET_naruto_eye.h5 -i data/test/sage_draw.jpeg
```

### Neural Networks :
---

I have used the basic neural networks for testing purpose and will add more.

#### LeNet :
---

Structure:

<p style="align: center">
<img src="/plots/LENET_schema.png" width="auto" height="500px"/>
</p>

Training result:

<p style="align: center">
<img src="/plots/LENET.png" width="50%"/>
</p>

#### Thanks for the References :
---

- https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
- https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
- https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
- https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
- https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
- Naruto fonts by [Cufonfonts](https://www.cufonfonts.com/font/ninja-naruto)
- [TensorFlow.js](https://js.tensorflow.org/)
- [Release Notes of Tensorflow.js](https://github.com/tensorflow/tfjs/releases)
- And All open source resources ..!