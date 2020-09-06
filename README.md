## U-GAT-IT &mdash; Paddle Implementation
### : Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

<div align="center">
  <img src="./assets/teaser.png">
</div>
## Usage

```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

### Train
```
> python main.py --dataset selfie2anime
```
* If the memory of gpu is **sufficient**, set `--light` to False

### Evaluation
```
> python main.py --dataset selfie2anime --phase eval
```
### Test A2B
```
> python main.py --dataset selfie2anime --phase test
```
### Train from a specific checkpoint<num1>
```
> python main.py --dataset selfie2anime --phase train --resume True --start_iteration <num1> --iteration <num2>
```

### Metric Result KID
we want to compare with official performance, so the same evaluation tool should be used.
It is a tensorflow-based script which will extract features from source images, target images and fake A2B images for final result. we need to organize folder in following format, and then run main.py
```
> python main.py
```
```
├── scoring
       ├── real_source
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── real_target
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── fake
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       ├── frechet_kernel_Inception_distance.py 
       ├── inception_score.py
       ├── main.py
       └── ...
```


### [Paper](https://arxiv.org/abs/1907.10830) | [Official Tensorflow code](https://github.com/taki0112/UGATIT)
The results of the paper came from the **Tensorflow code**


> **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
>
> **Abstract** *We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters.*


## Architecture
<div align="center">
  <img src = './assets/generator.png' width = '785px' height = '500px'>
</div>

---

<div align="center">
  <img src = './assets/discriminator.png' width = '785px' height = '450px'>
</div>

## Results
### Ablation study
<div align="center">
  <img src = './assets/ablation.png' width = '438px' height = '346px'>
</div>

### User study
<div align="center">
  <img src = './assets/user_study.png' width = '738px' height = '187px'>
</div>

### Comparison
<div align="center">
  <img src = './assets/kid.png' width = '787px' height = '344px'>
</div>
