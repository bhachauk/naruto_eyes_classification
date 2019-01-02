## Naruto Ocular Power Classifier using Neural Networks
---

<p style="align: center">
<img src="/data/naruto_eyes_prediction-web.gif" width="50%"/>
</p>

### Getting Started :
---

Classification on Naruto character eye images with various **CNN** for analysing the Neural Network accuracy.

###### Info :
---


|Classes| NeuralNetworks|
|-------|---------------|
|Sharingan|CNN1|
|Byakugan|CNN3|
|Sage|CNN4|
|Others|LeNet|
||VGG16|

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

I have used these neural networks. Still lot to go and include ... !

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

#### CNN4 :
---

Structure:

<p style="align: center">
<img src="/plots/CNN4_schema.png" width="auto" height="500px"/>
</p>

Training result:

<p style="align: center">
<img src="/plots/CNN4.png" width="auto" height="500px"/>
</p>


#### CNN3 :
---

Structure:

<p style="align: center">
<img src="/plots/CNN3_schema.png" width="auto" height="500px"/>
</p>

Training result:

<p style="align: center">
<img src="/plots/CNN3.png" width="auto" height="500px"/>
</p>

#### CNN1 :
---

Structure:

<p style="align: center">
<img src="/plots/CNN1_schema.png" width="auto" height="500px"/>
</p>

Training result:

<p style="align: center">
<img src="/plots/CNN1.png" width="50%"/>
</p>

#### Thanks for the References :
---

- https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
- https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
- https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
- https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
- https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
- Naruto fonts by [Cufonfonts](https://www.cufonfonts.com/font/ninja-naruto)
- And All open source resources ..!