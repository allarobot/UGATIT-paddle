import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Layer, Linear, Sequential, SpectralNorm
from paddle.fluid.layers import pad2d, instance_norm, relu, unsqueeze, tanh, adaptive_pool2d, spectral_norm
from paddle.fluid.layers import reduce_sum, reshape, concat, fill_constant, image_resize, leaky_relu
import numpy as np
from utils import ReflectionPad2D

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        # Downsampling
        self.pad2d1 = ReflectionPad2D(3)
        self.conv1 = Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=None)
        self.conv2 = Conv2D(ngf, ngf * 2, filter_size=3, stride=2, padding=0, bias_attr=None)
        self.conv3 = Conv2D(ngf * 2, ngf * 4, filter_size=3, stride=2, padding=0, bias_attr=None)

        n_downsampling = 2
        mult = 2**n_downsampling
        DownBlock = []
        self.pad2d2 = ReflectionPad2D(1)
        self.pad2d3 = ReflectionPad2D(1)
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(dim=(ngf * mult), use_bias=None)]
        # self.DownBlock1 = ResnetBlock(ngf * mult, None)

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=None)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=None)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=None, act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=None, act='relu')]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=None, act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=None, act='relu')]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=None)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=None)

        # Upsampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, None))

        # Upsampling
        self.convup1 = Conv2D(ngf * 4, ngf * 2, filter_size=3, stride=1, padding=0, bias_attr=None)
        self.iln1 = ILN(ngf * 2)
        self.convup2 = Conv2D(ngf * 2, ngf * 1, filter_size=3, stride=1, padding=0, bias_attr=None)
        self.iln2 = ILN(ngf * 1)
        self.convup3 = Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=None)


        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)

    def forward(self, input):
        x = self.pad2d1(input)
        x = self.conv1(x)
        x = instance_norm(x)
        x = relu(x)

        x = self.pad2d2(x)
        x = self.conv2(x)
        x = instance_norm(x)
        x = relu(x)

        x = self.pad2d3(x)
        x = self.conv3(x)
        x = instance_norm(x)
        x = relu(x)

        x = self.DownBlock(x)

        gap = adaptive_pool2d(x, pool_type='avg', pool_size=1)
        gap_logit = self.gap_fc(reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = unsqueeze(gap_weight, axes=2)
        gap_weight = unsqueeze(gap_weight, axes=3)
        gap = x * gap_weight

        gmp = adaptive_pool2d(x, pool_type='max', pool_size=1)
        gmp_logit = self.gmp_fc(reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = unsqueeze(gmp_weight, axes=2)
        gmp_weight = unsqueeze(gmp_weight, axes=3)
        gmp = x * gmp_weight

        cam_logit = concat([gap_logit, gmp_logit], 1)
        x = concat([gap, gmp], 1)
        x = relu(self.conv1x1(x))

        heatmap = reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = adaptive_pool2d(x, pool_size=1, pool_type='avg')
            x_ = self.FC(reshape(x_, shape=[x_.shape[0], -1]))
        else:
            x_ = self.FC(reshape(x, shape=[x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)

        x = image_resize(x, scale=2, resample='NEAREST')
        x = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        x = self.convup1(x)
        x = self.iln1(x)
        x = relu(x)

        x = image_resize(x, scale=2, resample='NEAREST')
        x = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        x = self.convup2(x)
        x = self.iln2(x)
        x = relu(x)

        x = pad2d(x, mode='reflect', paddings=[3, 3, 3, 3])
        x = self.convup3(x)
        out = tanh(x)

        return out, cam_logit, heatmap

class ResnetBlock(fluid.dygraph.Layer):

    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)

    def forward(self, x):
        instance = x
        out = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        out = self.conv1(out)
        out = instance_norm(out)
        out = relu(out)

        out = pad2d(out, mode='reflect', paddings=[1, 1, 1, 1])
        out = self.conv2(out)
        out = instance_norm(out)

        out = instance + out
        return out

class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)

        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        instance = x
        out = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = relu(out)

        out = pad2d(out, mode='reflect', paddings=[1, 1, 1, 1])
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return instance + out

class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        t = fluid.Tensor()
        self.eps = eps
        # self.rho = Layer.parameters(t.set(1, num_features, 1, 1))
        self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.ConstantInitializer(value=0.9))
        # self.rho = fill_constant(shape=self.rho.shape(), dtype='float32', value=0.9)
        # self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = np.mean(input, axis=[2, 3], keepdims=True), np.var(input, axis=[2, 3], keepdims=True)
        out_in = (input - in_mean) / np.sqrt(in_var + self.eps)
        ln_mean, ln_var = np.mean(input, axis=[1, 2, 3], keepdims=True), np.var(input, axis=[1, 2, 3], keepdims=True)
        out_ln = (input - ln_mean) / np.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln

        gamma = unsqueeze(gamma, axes=2)
        gamma = unsqueeze(gamma, axes=3)
        beta = unsqueeze(beta, axes=2)
        beta = unsqueeze(beta, axes=3)

        out = out * gamma + beta

        return out

class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        t = fluid.Tensor()
        self.eps = eps
        # self.rho = Layer.parameters(t.set(1, num_features, 1, 1))
        self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.ConstantInitializer(value=0.0))
        # self.gamma = Layer.parameters(t.set(1, num_features, 1, 1))
        self.gamma = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.ConstantInitializer(value=1.0))
        # self.beta = Layer.parameters(t.set(1, num_features, 1, 1))
        self.beta = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', default_initializer=fluid.initializer.ConstantInitializer(value=0.0))

        # self.rho = fill_constant(shape=self.rho.shape(), dtype='float32', value=0.0)
        # self.gamma = fill_constant(shape=self.gamma.shape(), dtype='float32', value=1.0)
        # self.beta = fill_constant(shape=self.beta.shape(), dtype='float32', value=0.0)
        # self.rho.data.fill_(0.0)
        # self.gamma.data.fill_(1.0)
        # self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = np.mean(input, axis=[2, 3], keepdims=True), np.var(input, axis=[2, 3], keepdims=True)
        out_in = (input - in_mean) / np.sqrt(in_var + self.eps)
        ln_mean, ln_var = np.mean(input, axis=[1, 2, 3], keepdims=True), np.var(input, axis=[1, 2, 3], keepdims=True)
        out_ln = (input - ln_mean) / np.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln

        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)
        self.conv2 = Conv2D(ndf, ndf * 2, filter_size=4, stride=2, padding=0, bias_attr=True)
        self.conv3 = Conv2D(ndf * 2, ndf * 4, filter_size=4, stride=2, padding=0, bias_attr=True)
        self.conv4 = Conv2D(ndf * 4, ndf * 8, filter_size=4, stride=1, padding=0, bias_attr=True)

        # Class Activation Map
        self.gap_fc = Linear(ndf * 8, 1, bias_attr=None)
        self.gmp_fc = Linear(ndf * 8, 1, bias_attr=None)
        self.conv1x1 = Conv2D(ndf * 16, ndf * 8, filter_size=1, stride=1, bias_attr=True)

        self.conv5 = Conv2D(ndf * 8, 1, filter_size=4, stride=1, padding=0, bias_attr=None)

    def forward(self, input):
        x = pad2d(input, mode='reflect', paddings=[1, 1, 1, 1])
        spectral_norm(self.conv1.parameters())
        x = self.conv1(x)
        x = leaky_relu(x, alpha=0.2)

        x = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        spectral_norm(self.conv2.parameters())
        x = self.conv2(x)
        x = leaky_relu(x, alpha=0.2)

        x = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        spectral_norm(self.conv3.parameters())
        x = self.conv3(x)
        x = leaky_relu(x, alpha=0.2)

        x = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        spectral_norm(self.conv4.parameters())
        x = self.conv4(x)
        x = leaky_relu(x, alpha=0.2)

        gap = adaptive_pool2d(x, pool_type='avg', pool_size=1)
        spectral_norm(self.gap_fc.parameters())
        gap_logit = self.gap_fc(reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = unsqueeze(gap_weight, axes=2)
        gap_weight = unsqueeze(gap_weight, axes=3)
        gap = x * gap_weight

        gmp = adaptive_pool2d(x, pool_type='avg', pool_size=1)
        spectral_norm(self.gmp_fc.parameters())
        gmp_logit = self.gmp_fc(reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = unsqueeze(gmp_weight, axes=2)
        gmp_weight = unsqueeze(gmp_weight, axes=3)
        gmp = x * gmp_weight

        cam_logit = concat([gap_logit, gmp_logit], 1)
        x = concat([gap, gmp], 1)
        x = leaky_relu(self.conv1x1(x))

        heatmap = reduce_sum(x, dim=1, keep_dim=True)

        x = pad2d(x, mode='reflect', paddings=[1, 1, 1, 1])
        spectral_norm(self.conv5.parameters())
        out = self.conv5(x)

        return out, cam_logit, heatmap

class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w










