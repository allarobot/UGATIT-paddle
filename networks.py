import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear,PRelu, SpectralNorm, LayerNorm, InstanceNorm
from paddle.fluid.dygraph import Sequential
import paddle.fluid.dygraph.nn as nn
#from paddle.fluid.Tensor import tensor
import numpy as np

w_param_attrs = fluid.ParamAttr(name="weight",
                                initializer = fluid.initializer.NormalInitializer(loc=0.0,scale=0.2),
                                learning_rate=0.0001,
                                regularizer=fluid.regularizer.L2Decay(0.0001),
                                trainable=True)
weight_init = fluid.initializer.NormalInitializer(loc=0.0,scale=0.2)  
weight_regularizer = fluid.regularizer.L2Decay(0.0001)

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        # ops for Down-Sampling 1/3
        DownBlock = []
        DownBlock += [
                      ReflectionPad2d(3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      ReLU()
                      ]
         
        # ops for Down-Sampling 2/3~3/3
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                          ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          ReLU()                        
                          ]

        # ops for Encoder Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # CAM of Generator
        self.gap_fc = Linear(ngf * mult, 1,bias_attr=False)  
        self.gmp_fc = Linear(ngf * mult, 1,bias_attr=False) 
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1,bias_attr=False,act='relu') #bias=True

        # ops for  Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult,bias_attr=False,act='relu'),
                  #nn.ReLU(True),
                  Linear(ngf * mult, ngf * mult,bias_attr=False,act='relu')
                  #nn.ReLU(True)
                  ]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  #nn.ReLU(True),
                  Linear(ngf * mult, ngf * mult,bias_attr=False, act='relu')
                  #nn.ReLU(True)
                  ]
            
        self.gamma = Linear(ngf * mult, ngf * mult,bias_attr=False)  # FC256
        self.beta = Linear(ngf * mult, ngf * mult,bias_attr=False) # FC256

        # ops for Decoder Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # ops for Decoder Up-Sampling
        n_upsampling = n_downsampling
        UpBlock2 = []
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            UpBlock2 += [#nn.Upsample(scale_factor=2, mode='nearest'),
                         #fluid.layers.pad2d(1),
                         Upsample(),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         LIN(int(ngf * mult / 2)),
                         ReLU()
                         ]
        
        UpBlock2 += [#fluid.layers.pad2d(3),
                     ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False,act='tanh')
                     #nn.Tanh()
                     ]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):  
        # Encoder and Bottleneck  
        # print("input.shape: ",input.shape)   
        x = self.DownBlock(input) # shape=(N,256,64,64)
        # print("x.shape: ",x.shape)
        # CAM 1/2
        gap = Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x) # shape=(N,256,1,1)
        gap = fluid.layers.reshape(gap, shape=[x.shape[0], -1]) #torch.Size([1, 1]) # shape=(N,256)
        gap_logit = self.gap_fc(gap) # shape=(N,1)
        gap_weight = list(self.gap_fc.parameters())[0] # shape=(256,1)
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0,3]) # shape=(1,256,1,1)
        gap = x * gap_weight    #shape=[N, 256, 64, 64]

        gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0,3])
        gmp = x * gmp_weight

        # Output_2
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)  ##Auxiliary classifier

        # CAM 2/2
        x = fluid.layers.concat([gap, gmp], 1)   #torch.Size([1, 512, 64, 64])      
        x = self.conv1x1(x) #torch.Size([1, 256, 64, 64]) 

        # Ouput_3
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  ## Attention map, #heatmap torch.Size([1, 1, 64, 64])

        # alpha, beta
        if self.light:
            # 1/3,shape(N,256,64,64) -->(N,256,1,1)
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg') # shape=(N,256,1,1)
            x_ = fluid.layers.reshape(x_, shape=[x.shape[0], -1]) # shape=(N,256)  
            # 2/3~3/3, x2 (N,256)-->(N,256) by FC256
            x_ = self.FC(x_)
        else:
            # 1/3 (N,256,64,64)-->(N,256*64*64)
            x_ = fluid.layers.reshape(x, shape=[x.shape[0], -1]) # shape=(N,256*64*64)
            # 2/3 (N,64*64*256)-->(N,256), 2/3 (N,256)-->(N,256) by FC
            x_ = self.FC(x_)
        # (N,256)-->(N,256), parameters for AdaILN
        gamma, beta = self.gamma(x_), self.beta(x_) # gamma torch.Size([N, 256]) beta torch.Size([N, 256])

        # Decoder bottleneck, after CAM layer with (alpah,beta)
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)

        # Decoder upsampling
        out = self.UpBlock2(x)

        #out torch.Size([1, 3, 256, 256]) cam_logit torch.Size([1, 2])  heatmap torch.Size([1, 1, 64, 64])
        return out, cam_logit, heatmap

class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU()                         
                       ]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)
                       ]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm1 = AdaILN(dim)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta)
        out= self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x
class AdaILN(fluid.dygraph.Layer):

    def __init__(self, in_channels, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho =self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.9))

    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input, gamma, beta):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho)*out_ln
        out=out*fluid.layers.unsqueeze(gamma,[2,3])+fluid.layers.unsqueeze(beta,[2,3])
        return out

class LIN(fluid.dygraph.Layer):

    def __init__(self, in_channels, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))  # 0.0 -- LN, 1.0 --IN
        self.gamma = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        
    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        # ex_rho = fluid.layers.expand(self.rho, expand_times = [input.shape[0], 1, 1, 1])
        # ex_gamma = fluid.layers.expand(self.gamma, expand_times = [input.shape[0], 1, 1, 1])
        # ex_beta = fluid.layers.expand(self.beta, expand_times = [input.shape[0], 1, 1, 1])
        # print("rho",fluid.layers.reduce_mean(self.rho).numpy())
        # print("gamma",fluid.layers.reduce_mean(self.gamma).numpy())
        # print("beta",fluid.layers.reduce_mean(self.beta).numpy())
        # out = ex_rho * out_in + (1 - ex_rho) * out_ln
        out = self.rho * out_in + (1 - self.rho)*out_ln
        # out=out*fluid.layers.unsqueeze(fluid.layers.unsqueeze(self.gamma,2),3)+fluid.layers.unsqueeze(fluid.layers.unsqueeze(self.beta,2),3)
        out = out * self.gamma + self.beta
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5): # input_nc = 3
        super(Discriminator, self).__init__()
        model = [
                 ReflectionPad2d(1),
                 Spectralnorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)),
                 Leaky_ReLU()
                ]        


        for i in range(1, n_layers - 1):
            mult = 2 ** (i - 1)
            model += [
                      #fluid.layers.pad2d(1),
                      ReflectionPad2d(1),
                      Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True)),
                      Leaky_ReLU()
                      ]        

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)

        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))   

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input) #[1, 2048, 8, 8] / [1, 512, 32, 32]

        #gap = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
        gap =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x) #[1, 2048, 1, 1]
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) 
        gap_logit = self.gap_fc(gap)#torch.Size([1, 1])
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0,3]) 
        gap = x * gap_weight #[1, 2048, 2, 2]

        #gmp = fluid.layers.adaptive_pool2d(x, 1,pool_type='max')
        gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])        
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0,3])
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x= fluid.layers.leaky_relu(self.conv1x1(x),alpha=0.2)

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap

class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale)
        return out

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

class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out
        
        
class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")
    
        
class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace=inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            y=fluid.layers.relu(x)
            return y
        
    
class Leaky_ReLU(fluid.Layer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.leaky_relu = lambda x: fluid.layers.leaky_relu(x, alpha=alpha)

    def forward(self, x):
        return self.leaky_relu(x)
