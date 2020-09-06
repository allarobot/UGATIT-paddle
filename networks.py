import paddle.fluid as fluid
#from paddle.fluid.layer_helper import LayerHelper
#from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear, PRelu, LayerNorm, InstanceNorm#,SpectralNorm
from paddle.fluid.dygraph import Sequential
#import paddle.fluid.dygraph.nn as nn
import numpy as np
from ops import *

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
                      Conv2D_(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0),
                      InstanceNorm(ngf), #center,scale
                      ReLU()
                      ]
         
        # ops for Down-Sampling 2/3~3/3
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                          ReflectionPad2d(1),
                          Conv2D_(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0),
                          InstanceNorm(ngf * mult * 2),
                          ReLU()                        
                          ]

        # ops for Encoder Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=True)]

        # CAM of Generator
        self.gap_fc = Linear_(ngf * mult, 1)  
        self.gmp_fc = Linear_(ngf * mult, 1) 
        self.conv1x1 = Conv2D_(ngf * mult * 2, ngf * mult, filter_size=1, stride=1,act='relu')

        ## ops for  Gamma, Beta block
        # if self.light:
        #     FC = [Linear(ngf * mult, ngf * mult,bias_attr=False,act='relu'),
        #           #nn.ReLU(True),
        #           Linear(ngf * mult, ngf * mult,bias_attr=False,act='relu')
        #           #nn.ReLU(True)
        #           ]
        # else:
        #     FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False,act='relu'),
        #           #nn.ReLU(True),
        #           Linear(ngf * mult, ngf * mult,bias_attr=False, act='relu')
        #           #nn.ReLU(True)
        #           ]
        # self.gamma = Linear(ngf * mult, ngf * mult,bias_attr=False)  # FC256
        # self.beta = Linear(ngf * mult, ngf * mult,bias_attr=False) # FC256
        if self.light:
            self.mlp = MLP(ngf * mult,ngf * mult,light=True)
        else:
            self.mlp = MLP(img_size // mult * img_size // mult * ngf * mult, ngf * mult,light=False)

        # ops for Decoder Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult))

        # ops for Decoder Up-Sampling
        n_upsampling = n_downsampling
        UpBlock2 = []
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            UpBlock2 += [
                         Upsample(),
                         ReflectionPad2d(1),
                         Conv2D_(ngf * mult, ngf * mult // 2, filter_size=3, stride=1, padding=0),
                         LIN(ngf * mult // 2),
                         ReLU()
                         ]
        
        UpBlock2 += [#fluid.layers.pad2d(3),
                     ReflectionPad2d(3),
                     Conv2D_(ngf, output_nc, filter_size=7, stride=1, padding=0, act='tanh')
                     #nn.Tanh()
                     ]

        self.DownBlock = Sequential(*DownBlock)
        #self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):  
        # Encoder and Bottleneck  
        x = self.DownBlock(input) # shape=(N,256,64,64)

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


        gamma,beta = self.mlp(x)

        # Decoder bottleneck, after CAM layer with (alpah,beta)
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)

        # Decoder upsampling
        x = self.UpBlock2(x)

        #out torch.Size([1, 3, 256, 256]) cam_logit torch.Size([1, 2])  heatmap torch.Size([1, 1, 64, 64])
        return x, cam_logit, heatmap


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5): # input_nc = 5/7
        super(Discriminator, self).__init__()
        encoder = [
                 ReflectionPad2d(1),
                 Spectralnorm(Conv2D_(input_nc, ndf, filter_size=4, stride=2, padding=0),dim=1),
                 Leaky_ReLU()
                ]        

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            encoder += [
                      ReflectionPad2d(1),
                      Spectralnorm(Conv2D_(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0),dim=1),
                      Leaky_ReLU()
                      ]        
        mult = 2**(n_layers-2-1)
        encoder += [
                    ReflectionPad2d(1),
                    Spectralnorm(Conv2D_(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0),dim=1),
                    Leaky_ReLU()
                    ] 
        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear_(ndf * mult, 1),dim=0)
        self.gmp_fc = Spectralnorm(Linear_(ndf * mult, 1),dim=0)
        self.conv1x1 = Conv2D_(ndf * mult * 2, ndf * mult, filter_size=1, stride=1)

        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D_(ndf * mult, 1, filter_size=4, stride=1, padding=0),dim=1)  

        self.encoder = Sequential(*encoder)

    def forward(self, input):
        x = self.encoder(input) #[1, 2048, 8, 8] / [1, 512, 32, 32]

        #gap = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
        gap =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x) #[1, 2048, 1, 1]
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) 
        gap_logit = self.gap_fc(gap)#torch.Size([1, 1])
        gap_weight = list(self.gap_fc.parameters())[0]
        #print("gap_fc param: ",self.gap_fc.parameters())
        #print("shape of gap_weight:",gap_weight.shape)
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

        x = self.conv1x1(x)
        x= fluid.layers.leaky_relu(x,alpha=0.2)

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap