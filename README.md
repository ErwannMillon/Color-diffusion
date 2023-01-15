# Color Diffusion

## Overview
This project is a simple example of how we can use diffusion models to colorize black and white images. 

This implementation uses the LAB color space, a 3 channel alternative to the RGB color space. 
The "L" (Lightness) channel in this space is equivalent to a greyscale image: it represents the luminous intensity of each pixel. The two other channels are used to represent the color of each pixel. 

To train the model, we first load color images and convert them to LAB.
Then, we add noise only to the color channels, keeping the L channel constant. The model gets this channel "for free" because it doesn't need to learn how to predict the greyscale image: it is always known at train and test time. 

The model is a UNet that takes a 3 channel LAB input (the ground-truth greyscale channel concatenated with noised AB channels) and outputs a 2 channel prediction of the color noise. 

In addition to receiving the greyscale channel as input, the UNet is also conditioned on features extracted from the greyscale channel. Intermediate feature maps from an encoder (implemented as the first half of a UNet) are concatenated with the features of the main denoising UNet throughout the downsampling stage of the forward pass.

This was just a quick proof of concept to satisfy my curiosity and get a feel for training diffusion models from scratch, so the results are really basic. There are many ways this project could be improved, such as:
- Using pretrained face recognition networks like ArcFace or FaceNet as feature extractors to get the conditioning features
- Implementing cross attention on the embeddings
- Pretraining the greyscale feature extractor as the encoder stage of a greyscale autoencoder

A lot of code for the dataset and LAB color operations was adapted from moein-sharitania's colorization project, which used Conditional GANs
https://github.com/moein-shariatnia/Deep-Learning

I implemented optional dynamic thresholding as in Assembly AI's Minimagen project (the Assembly AI blog posts are excellent for getting a deep understanding of the maths and concepts behind diffusion models)
https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/

The UNet architecture was adapted from denoising-diffusion-pytorch
https://github.com/lucidrains/denoising-diffusion-pytorch

## Usage
Use inference.py for command line colorization. See help for more info on the configuration flags you can pass
`
python inference.py --image-path <IMG_PATH> --checkpoint <CKPT_PATH> --output <OUTPUT_PATH>
`

Or run 
`
python app.py for a simple gradio web UI
`



