#Deep Learning Course Project: Conditional PixelCNN++ for Image Classification 

**The goal of this project was to implement the Conditional PixelCNN++ model and train it on the given dataset.** After that, the model can both generate new images and classify the given images. 
link: https://github.com/DSL-Lab/CPEN455HW-2023W2


PixelCNN++ is a powerful generative model with tractable likelihood. It models the joint distribution of pixels over an image x as the following product of conditional distributions.

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/-jZg8HEMyFnpduNsi-Alt.png" width = "500" align="center"/>

where x_i is a single pixel.

Given a class embedding c, PixelCNN++ can be extended to conditional generative tasks following:

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/_jv7O2Z_1s1oYLXjIqS1V.png" width = "260" align="center"/>

In this case, with a trained conditional PixelCNN++, we could directly apply it to the zero-shot image classification task by:

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/P4co1MxbW8tmhgYwBNOxk.png" width = "350" align="center"/>

The aim of this project was:
* Adapting the given source code to perform conditional image generation.
  
* Conditionally generate images and evaluate the generated images using FID score.

* Convert the output of conditional PixelCNN++ to the prediction labels when given a new image.
  


## Original PixelCNN++ code
You need to install the required packages by running the following command:
```
conda create -n cpen455 python=3.10.13
conda activate cpen455
conda install pip3
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

Pixelcnn++ original paper: https://arxiv.org/abs/1701.05517

And there are some repositories that implement the PixelCNN++ model. You can find them in the following link:

1. Original PixelCNN++ repository implemented by OpenAI: https://github.com/openai/pixel-cnn

2. Pytorch implementation of PixelCNN++: https://github.com/pclucas14/pixel-cnn-pp
