import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear,PRelu, SpectralNorm, LayerNorm, InstanceNorm
from paddle.fluid.dygraph import Sequential
import paddle.fluid.dygraph.nn as nn
#from paddle.fluid.Tensor import tensor
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
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=True),
                      InstanceNorm(ngf), #center,scale
                      ReLU()
                      ]
         
        # ops for Down-Sampling 2/3~3/3
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                          ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=True),
                          InstanceNorm(ngf * mult * 2),
                          ReLU()                        
                          ]

        # ops for Encoder Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=True)]

        # CAM of Generator
        self.gap_fc = Linear(ngf * mult, 1,bias_attr=True)  
        self.gmp_fc = Linear(ngf * mult, 1,bias_attr=True) 
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1,bias_attr=True,act='relu') #bias=True

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
            self.mlp = MLP(ngf * mult,ngf * mult)
        else:
            self.mlp = MLP(img_size // mult * img_size // mult * ngf * mult, ngf * mult)

        # ops for Decoder Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult))

        # ops for Decoder Up-Sampling
        n_upsampling = n_downsampling
        UpBlock2 = []
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            UpBlock2 += [#nn.Upsample(scale_factor=2, mode='nearest'),
                         #fluid.layers.pad2d(1),
                         Upsample(),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, ngf * mult // 2, filter_size=3, stride=1, padding=0, bias_attr=True),
                         LIN(ngf * mult // 2),
                         ReLU()
                         ]
        
        UpBlock2 += [#fluid.layers.pad2d(3),
                     ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=True,act='tanh')
                     #nn.Tanh()
                     ]

        self.DownBlock = Sequential(*DownBlock)
        #self.FC = Sequential(*FC)
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

        # # alpha, beta
        # if self.light:
        #     # 1/3,shape(N,256,64,64) -->(N,256,1,1)
        #     x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg') # shape=(N,256,1,1)
        #     x_ = fluid.layers.reshape(x_, shape=[x.shape[0], -1]) # shape=(N,256)  
        #     # 2/3~3/3, x2 (N,256)-->(N,256) by FC256
        #     x_ = self.FC(x_)
        # else:
        #     # 1/3 (N,256,64,64)-->(N,256*64*64)
        #     x_ = fluid.layers.reshape(x, shape=[x.shape[0], -1]) # shape=(N,256*64*64)
        #     # 2/3 (N,64*64*256)-->(N,256), 2/3 (N,256)-->(N,256) by FC
        #     x_ = self.FC(x_)
        # (N,256)-->(N,256), parameters for AdaILN
        #gamma, beta = self.gamma(x_), self.beta(x_) # gamma torch.Size([N, 256]) beta torch.Size([N, 256])
        gamma,beta = self.mlp(x)

        # Decoder bottleneck, after CAM layer with (alpah,beta)
        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)

        # Decoder upsampling
        x = self.UpBlock2(x)

        #out torch.Size([1, 3, 256, 256]) cam_logit torch.Size([1, 2])  heatmap torch.Size([1, 1, 64, 64])
        return x, cam_logit, heatmap




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