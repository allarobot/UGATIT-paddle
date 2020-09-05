import time, itertools
import paddle.fluid as fluid
import paddle
from networks import *
from utils import *
from glob import glob

from dataset import ImageFolder
from paddlex.cls import transforms
from paddle.fluid.io import DataLoader

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.log_path = f'{self.result_dir}/log.txt'
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        ensure_folder_exsit(self.result_dir)

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    def log(self, message):
        #if self.config.verbose:
        print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
    ##################################################################################
    # Model
    ##################################################################################


    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ResizeByShort(short_size=self.img_size + 30, max_size=-1),
            transforms.RandomCrop(self.img_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.ResizeByShort(short_size=self.img_size, max_size=-1),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), tranform=train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), tranform=train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), tranform=test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), tranform=test_transform)

        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True, return_list=True, places=fluid.cuda_places(0))
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True, return_list=True, places=fluid.cuda_places(0))
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False, return_list=True, places=fluid.cuda_places(0))
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False, return_list=True, places=fluid.cuda_places(0))

        #
        # self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'))
        # self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'))
        # self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'))
        # self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'))
 
        # self.trainA_loader = self.trainA
        # self.trainB_loader = self.trainB
        # self.testA_loader = self.testA
        # self.testB_loader = self.testB        
        

        #""" Define Generator, Discriminator """
        #self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        #self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        #self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        #self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        #self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        #self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        #""" Define Loss """
        ##self.L1_loss = nn.L1Loss().to(self.device)
        ##self.MSE_loss = nn.MSELoss().to(self.device)
        ##self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        #self.L1_loss =  fluid.dygraph.L1Loss()
        #self.MSE_loss = fluid.dygraph.MSELoss()
        #self.BCE_loss = fluid.dygraph.BCELoss()       

        #""" Trainer """
        ##self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        ##self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        #self.G_optim  = fluid.optimizer.Adam(learning_rate=self.lr, parameter_list=itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()))
        #self.D_optim  = fluid.optimizer.Adam(learning_rate=self.lr, parameter_list=itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()))
        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        
        place = fluid.CUDAPlace(0) if self.device=='cuda' else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            
            """ Define Generator, Discriminator """
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
            """ Define Loss """
            #self.L1_loss = nn.L1Loss().to(self.device)
            #self.MSE_loss = nn.MSELoss().to(self.device)
            #self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
            self.L1_loss =  fluid.dygraph.L1Loss()
            self.MSE_loss = fluid.dygraph.MSELoss()
            self.BCE_loss = BCEWithLogitsLoss()       
    
            """ Trainer """
            #self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
            #self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
            # self.G_optim  = fluid.optimizer.Adam(learning_rate=self.lr, parameter_list=itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()))
            # self.D_optim  = fluid.optimizer.Adam(learning_rate=self.lr, parameter_list=itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()))            
            # self.G_optim  = fluid.optimizer.Adam(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            # self.D_optim  = fluid.optimizer.Adam(learning_rate=self.lr, beta1=0.5, beta2=0.999,  parameter_list=itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()),regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))  
            self.G_optim  = fluid.optimizer.Adam(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=self.genA2B.parameters()+self.genB2A.parameters(),regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            
            self.D_optim  = fluid.optimizer.Adam(learning_rate=self.lr, beta1=0.5, beta2=0.999,  parameter_list=self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters(),regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))    
            start_iter = 4391
            if self.resume:
                print('load model!!!')
                self.sload()
                # model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
                # if not len(model_list) == 0:
                #     model_list.sort()
                #     start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                #     self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                #     print(" [*] Load SUCCESS")
                #     # if self.decay_flag and start_iter > (self.iteration // 2):
                #     #     self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                #     #     self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
        
            # training loop
            print('training start !')
            start_time = time.time()
            for step in range(start_iter, self.iteration + 1):
                if self.decay_flag and step > (self.iteration // 2):
                    self.G_optim.param_and_grad[0]['learning_rate'] -= (self.lr / (self.iteration // 2))
                    self.D_optim.param_and_grad[0]['learning_rate'] -= (self.lr / (self.iteration // 2))
        
                try:
                    #real_A, _ = trainA_iter.next()
                    real_A, _ = next(trainA_iter)
                except:
                    trainA_iter = iter(self.trainA_loader)
                    real_A, _ = next(trainA_iter)
        
                try:
                    real_B, _ = next(trainB_iter)
                except:
                    trainB_iter = iter(self.trainB_loader)
                    real_B, _ = next(trainB_iter)
                    
                real_A = fluid.dygraph.base.to_variable(real_A)
                real_B = fluid.dygraph.base.to_variable(real_B)
              
                #for trainA,trainB in zip(self.trainA_loader,self.trainB_loader):
                    ##real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                #A2B = np.zeros((self.img_size * 7, 0, 3))
                #B2A = np.zeros((self.img_size * 7, 0, 3))   
                
                #fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                #fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                #fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                #fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                #fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                #fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                #A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                           #cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                           #RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                           #cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                           #RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                           #cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                           #RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                #B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                           #cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                           #RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                           #cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                           #RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                           #cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                           #RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)
    
                #cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                #cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)                
                #real_A,_=trainA
                #real_B,_=trainB
                # Update D
                # self.D_optim.clear_gradients()
        
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)
        
                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)
        
                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                
                ones = fluid.dygraph.to_variable(np.ones(real_GA_logit.shape).astype('float32'))
                zeros = fluid.dygraph.to_variable(np.zeros(fake_GA_logit.shape).astype('float32'))
                D_ad_loss_GA = self.MSE_loss(real_GA_logit, ones) + self.MSE_loss(fake_GA_logit, zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_GA_cam_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_GA_cam_logit.shape).astype('float32'))                
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, ones) + self.MSE_loss(fake_GA_cam_logit,zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_LA_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_LA_logit.shape).astype('float32'))                   
                D_ad_loss_LA = self.MSE_loss(real_LA_logit,ones) + self.MSE_loss(fake_LA_logit, zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_LA_cam_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_LA_cam_logit.shape).astype('float32'))                 
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, ones) + self.MSE_loss(fake_LA_cam_logit,zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_GB_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_GB_logit.shape).astype('float32'))                    
                D_ad_loss_GB = self.MSE_loss(real_GB_logit,ones) + self.MSE_loss(fake_GB_logit, zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_GB_cam_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_GB_cam_logit.shape).astype('float32'))                 
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit,ones) + self.MSE_loss(fake_GB_cam_logit,zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_LB_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_LB_logit.shape).astype('float32'))                 
                D_ad_loss_LB = self.MSE_loss(real_LB_logit,ones) + self.MSE_loss(fake_LB_logit,zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(real_LB_cam_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_LB_cam_logit.shape).astype('float32'))                 
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,ones) + self.MSE_loss(fake_LB_cam_logit,zeros)
        
                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)
        
                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.D_optim.minimize(Discriminator_loss)
                self.disGA.clear_gradients()
                self.disLA.clear_gradients()
                self.disGB.clear_gradients()
                self.disLB.clear_gradients()
                self.genA2B.clear_gradients()
                self.genB2A.clear_gradients()
                #self.D_optim.step()
        
                # Update G
                # self.G_optim.clear_gradients()
        
                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
        
                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)
        
                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)
        
                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
        
                ones = fluid.dygraph.to_variable(np.ones(fake_GA_logit.shape).astype('float32'))                  
                G_ad_loss_GA = self.MSE_loss(fake_GA_logit,ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_GA_cam_logit.shape).astype('float32')) 
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit,ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_LA_logit.shape).astype('float32')) 
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit,ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_LA_cam_logit.shape).astype('float32')) 
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_GB_logit.shape).astype('float32')) 
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit,ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_GB_cam_logit.shape).astype('float32')) 
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit,ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_LB_logit.shape).astype('float32')) 
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit,ones)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_LB_cam_logit.shape).astype('float32')) 
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit,ones)
        
                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)
        
                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)
        
                ones = fluid.dygraph.to_variable(np.ones(fake_B2A_cam_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_A2A_cam_logit.shape).astype('float32'))         
                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,ones) + self.BCE_loss(fake_A2A_cam_logit,zeros)
                
                ones = fluid.dygraph.to_variable(np.ones(fake_A2B_cam_logit.shape).astype('float32'))
                zeros=fluid.dygraph.to_variable(np.zeros(fake_B2B_cam_logit.shape).astype('float32'))                  
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,ones) + self.BCE_loss(fake_B2B_cam_logit,zeros)
        
                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B
        
                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_optim.minimize(Generator_loss)
                self.disGA.clear_gradients()
                self.disLA.clear_gradients()
                self.disGB.clear_gradients()
                self.disLB.clear_gradients()
                self.genA2B.clear_gradients()
                self.genB2A.clear_gradients()                
                #self.G_optim.step()
        
                # # clip parameter of AdaILN and ILN, applied after optimizer step
                # def clip_rho(net, vmin=0, vmax=1):
                #     for name, param in net.named_parameters():
                #         if 'rho' in name:
                #             param.set_value(fluid.layers.clip(param, vmin, vmax))    
                # clip_rho(self.genA2B)
                # clip_rho(self.genB2A)


                message = "[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss)
                self.log(message)

                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))
        
                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = next(trainA_iter)
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = next(trainA_iter)
        
                        try:
                            real_B, _ = next(trainB_iter)
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = next(trainB_iter)
                        real_A = fluid.dygraph.to_variable(real_A)
                        real_B = fluid.dygraph.to_variable(real_B)                    
                        #real_A, real_B = real_A.to(self.device), real_B.to(self.device)
        
                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
        
                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
        
                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
        
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)
        
                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)
        
                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = next(testA_iter)
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = next(testA_iter)
        
                        try:
                            real_B, _ = next(testB_iter)
                        except:
                            testB_iter = iter(self.testB_loader)
                            real_B, _ = next(testB_iter)  
                        real_A = fluid.dygraph.to_variable(real_A)
                        real_B = fluid.dygraph.to_variable(real_B)                        
                        #real_A, real_B = real_A.to(self.device), real_B.to(self.device)
        
                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
        
                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
        
                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
        
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)
        
                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)
                    ensure_folder_exsit(os.path.join(self.result_dir, self.dataset, 'img'))
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        
                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
        
                if step % 5000 == 0:
                    pa=os.path.join(self.result_dir, self.dataset ,'genA2B_'+str(step))
                    fluid.save_dygraph(self.genA2B.state_dict(), pa)
                    pb=os.path.join(self.result_dir, self.dataset , 'genB2A_'+str(step))
                    fluid.save_dygraph(self.genB2A.state_dict(), pb)
                    pc=os.path.join(self.result_dir, self.dataset ,'genA2B_'+str(step))
                    fluid.save_dygraph(self.G_optim.state_dict(),pc)
                    pd=os.path.join(self.result_dir, self.dataset , 'disGA_'+str(step))
                    fluid.save_dygraph(self.disGA.state_dict(), pd)
                    pe=os.path.join(self.result_dir, self.dataset , 'disGB_'+str(step))
                    fluid.save_dygraph(self.disGB.state_dict(), pe)
                    pf=os.path.join(self.result_dir, self.dataset , 'disLA_'+str(step))
                    fluid.save_dygraph(self.disLA.state_dict(), pf)   
                    pg=os.path.join(self.result_dir, self.dataset , 'disLB_'+str(step))
                    fluid.save_dygraph(self.disLB.state_dict(), pg)
                    ph=os.path.join(self.result_dir, self.dataset , 'genB2A_'+str(step))
                    fluid.save_dygraph(self.D_optim.state_dict(), ph)                      
                                    

    def save(self, dir, step):
        ensure_folder_exsit(dir)
        pa=os.path.join(dir, self.dataset + 'genA2B_'+str(step))
        fluid.save_dygraph(self.genA2B.state_dict(), pa)
        
        pb=os.path.join(dir, self.dataset + 'genB2A_'+str(step))
        fluid.save_dygraph(self.genB2A.state_dict(), pb)
        
        pc=os.path.join(dir, self.dataset + 'genA2B_'+str(step))
        fluid.save_dygraph(self.G_optim.state_dict(), pc)
        
        pd=os.path.join(dir, self.dataset + 'disGA_'+str(step))
        fluid.save_dygraph(self.disGA.state_dict(),pd)
        
        pe=os.path.join(dir, self.dataset + 'disGB_'+str(step))
        fluid.save_dygraph(self.disGB.state_dict(), pe)
        
        pf=os.path.join(dir, self.dataset + 'disLA_'+str(step))
        fluid.save_dygraph(self.disLA.state_dict(), pf)   
        
        pg=os.path.join(dir, self.dataset + 'disLB_'+str(step))
        fluid.save_dygraph(self.disLB.state_dict(), pg)
        
        ph=os.path.join(dir, self.dataset + 'genB2A_'+str(step))
        fluid.save_dygraph(self.D_optim.state_dict(), ph)          

    def sload(self):
        pa='results/selfie2anime/smodel/genA2B_30000.pdparams'
        a,c=fluid.load_dygraph(pa)
        
        pb='results/selfie2anime/smodel/genB2A_30000.pdparams'
        b,h=fluid.load_dygraph(pb)
        
        # pc='results/selfie2anime/smodel/G_optim_30000.pdopt'
        # _,c=fluid.load_dygraph(pc)
        
        pd='results/selfie2anime/smodel/disGA_30000.pdparams'
        d,_=fluid.load_dygraph(pd)
        
        pe='results/selfie2anime/smodel/disGB_30000.pdparams'
        e,_=fluid.load_dygraph(pe)
        
        pf='results/selfie2anime/smodel/disLA_30000.pdparams'
        f,_=fluid.load_dygraph(pf)   
        
        pg='results/selfie2anime/smodel/disLB_30000.pdparams'
        g,_=fluid.load_dygraph(pg)
        
        # ph='results/selfie2anime/smodel/D_optim_30000.pdopt'
        # _,h=fluid.load_dygraph(ph)  
        self.genA2B.load_dict(a)
        self.genB2A.load_dict(b)
        self.G_optim.set_dict(c)
        self.disGA.load_dict(d)
        self.disGB.load_dict(e)
        self.disLA.load_dict(f)
        self.disLB.load_dict(g)        
        self.D_optim.set_dict(h)        
        #params = fluid.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        #self.genA2B.load_state_dict(params['genA2B'])
        #self.genB2A.load_state_dict(params['genB2A'])
        #self.disGA.load_state_dict(params['disGA'])
        #self.disGB.load_state_dict(params['disGB'])
        #self.disLA.load_state_dict(params['disLA'])
        #self.disLB.load_state_dict(params['disLB'])

    def load(self, dir, step):
        pa=os.path.join(dir, self.dataset + 'genA2B_'+str(step))
        a,c=fluid.load_dygraph(pa)
        
        pb=os.path.join(dir, self.dataset + 'genB2A_'+str(step))
        b,h=fluid.load_dygraph(pb)
        
        # pc=os.path.join(dir, self.dataset + 'G_optim_'+str(step))
        # _,c=fluid.load_dygraph(pc)
        
        pd=os.path.join(dir, self.dataset + 'disGA_'+str(step))
        d,_=fluid.load_dygraph(pd)
        
        pe=os.path.join(dir, self.dataset + 'disGB_'+str(step))
        e,_=fluid.load_dygraph(pe)
        
        pf=os.path.join(dir, self.dataset + 'disLA_'+str(step))
        f,_=fluid.load_dygraph(pf)   
        
        pg=os.path.join(dir, self.dataset + 'disLB_'+str(step))
        g,_=fluid.load_dygraph(pg)
        
        # ph=os.path.join(dir, self.dataset + 'D_optim_'+str(step))
        # _,h=fluid.load_dygraph(ph)  
        self.genA2B.load_dict(a)
        self.genB2A.load_dict(b)
        self.G_optim.set_dict(c)
        self.disGA.load_dict(d)
        self.disGB.load_dict(e)
        self.disLA.load_dict(f)
        self.disLB.load_dict(g)        
        self.D_optim.set_dict(h)        
        #params = fluid.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        #self.genA2B.load_state_dict(params['genA2B'])
        #self.genB2A.load_state_dict(params['genB2A'])
        #self.disGA.load_state_dict(params['disGA'])
        #self.disGB.load_state_dict(params['disGB'])
        #self.disLA.load_state_dict(params['disLA'])
        #self.disLB.load_state_dict(params['disLB'])

    def test_load(self):
        pa='results\\selfie2anime\\model\\genA2B_20000.pdparams'
        a,_=fluid.load_dygraph(pa)
        
        pb='results\\selfie2anime\\model\\genB2A_20000.pdparams'
        b,_=fluid.load_dygraph(pb)
        
        #pc=os.path.join(dir, self.dataset + 'G_optim_'+str(step))
        #c=fluid.load_dygraph(self.G_optim.state_dict(),pc)
        
        #pd=os.path.join(dir, self.dataset + 'disGA_'+str(step))
        #d=fluid.load_dygraph(self.disGA.state_dict(), pd)
        
        #pe=os.path.join(dir, self.dataset + 'disGB_'+str(step))
        #e=fluid.load_dygraph(self.disGB.state_dict(), pe)
        
        #pf=os.path.join(dir, self.dataset + 'disLA_'+str(step))
        #f=fluid.load_dygraph(self.disLA.state_dict(), pf)   
        
        #pg=os.path.join(dir, self.dataset + 'disLB_'+str(step))
        #g=fluid.load_dygraph(self.disLB.state_dict(), pg)
        
        #ph=os.path.join(dir, self.dataset + 'D_optim_'+str(step))
        #h=fluid.load_dygraph(self.D_optim.state_dict(), ph)  
        self.genA2B.load_dict(a)
        self.genB2A.load_dict(b)
        #self.G_optim.set_dict(c)
        #self.disGA.load_dict(d)
        #self.disGB.load_dict(e)
        #self.disLA.load_dict(f)
        #self.disLB.load_dict(g)        
        #self.D_optim.set_dict(h)  
    def test(self):
        place = fluid.CUDAPlace(0) if self.device=='cuda' else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            
            """ Define Generator, Discriminator """
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            # self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            # self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            # self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            # self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)  
            # self.G_optim  = fluid.optimizer.Adam(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=self.genA2B.parameters()+self.genB2A.parameters(),regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            
            # self.D_optim  = fluid.optimizer.Adam(learning_rate=self.lr, beta1=0.5, beta2=0.999,  parameter_list=self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters(),regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))               
            
            #model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
            #if not len(model_list) == 0:
                #model_list.sort()
                #iter = int(model_list[-1].split('_')[-1].split('.')[0])
                #self.test_load()
                #print(" [*] Load SUCCESS")
            #else:
                #print(" [*] Load FAILURE")
                #return
            self.test_load()
            self.genA2B.eval(), self.genB2A.eval()
            ensure_folder_exsit(os.path.join(self.result_dir, self.dataset, 'test'))
            for n, (real_A, _) in enumerate(self.testA_loader):
                if n>6:
                    break
                real_A = fluid.dygraph.to_variable(real_A)
    
                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
    
                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
    
                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
    
                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                      cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                      cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                      cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)
    
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)
    
            for n, (real_B, _) in enumerate(self.testB_loader):
                if n>6:
                    break                
                real_B = fluid.dygraph.to_variable(real_B)
    
                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)
    
                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)
    
                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
    
                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                      cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                      cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                      cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)
    
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
