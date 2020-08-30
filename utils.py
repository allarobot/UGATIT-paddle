import os, cv2
import paddle.fluid as fluid
from paddle.fluid.layers import unsqueeze
from scipy import misc
import numpy as np

class ReflectionPad2D(fluid.dygraph.Layer):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, mode='reflect')



def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x / 127.5 - 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images + 1.0) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j : h * (j + 1), w * i : w * (i + 1), :] = image

    return img

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.unit8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    t = fluid.Tensor()

    mean = t.set(mean, fluid.dygraph.to_variable())
    mean = unsqueeze(mean, axes=0)
    mean = unsqueeze(mean, axes=2)
    mean = unsqueeze(mean, axes=3)

    std = t.set(std, fluid.dygraph.to_variable())
    std = unsqueeze(std, axes=0)
    std = unsqueeze(std, axes=2)
    std = unsqueeze(std, axes=3)

    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cup().numpy().transpose(1, 2, 0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)



