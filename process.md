_*input*_: 100 LGE MRI data with ground truth labels. 
    The original resolution of these data is 0.625*0.625*0.625 mm3
    47 of them are with 576*576*88 voxels and 53 of them are with 640*640*88 voxels, -> 576*576*44
1. estimate a fixed size of region that covers the whole artiral
2. cropping the volume according to the estimate:
    origin -> 576*576*96
    3D CLAHE: computing localized hitogram over different region, for enhancing the contraction of local details # MRI数据增强
    sample-wise normalization: each volumn is substracted by the mean value of intensity and divided by the derivation of intensity.
3. segmentation training:
    network architecture:
        Name: fully convolutional neural network
        Derivation: V-Net
        Network:   residual vnet
            left: encode: each layer has some steps:
                1. feature extraction using 5*5*5 volumetric convolutional core(this step 1-2 times per layer)
                2. element-wise suming the feature extracted and the raw input of this layer.
                3. Spontaneously, feed the sum to the right side of the network.
                4. down-pooling using 2*2*2 volumetric convolutional core, with stride of 2.
                5. use PRelu for batch normalization
            right: decode: predicting the probability of backgroud and foreground

F(x)
new : F(x) = 1+H(x)
z = f(x)
x = g(t)
z 
vgg19
res

prediction -> 576*576*88 or 640*640*88