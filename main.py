import numpy as np 
import matplotlib.pyplot as plt 
from stylegan_layers import  G_mapping,G_synthesis
from read_image import image_reader
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
from perceptual_model import VGG16_for_Perceptual
import torch.optim as optim
import os,random
from tqdm import tqdm
import networks
import itertools
from torchvision import transforms, utils
try:
    from networks.resample2d_package.resample2d import Resample2d
except:
    from .networks.resample2d_package.resample2d import Resample2d

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def make_dir(path):
    if os.path.isdir(path)==False:
        os.makedirs(path)
    else:
        pass


def main():
    parser = argparse.ArgumentParser(description='Find latent representation with consecutive images')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution',default=1024,type=int)
    parser.add_argument('--weight_file',default="weight_files/karras2019stylegan-ffhq-1024x1024.pt",type=str)
    parser.add_argument('--iteration',default=5000,type=int)
    parser.add_argument('--undex', type=int, default=1)
    parser.add_argument('--frames', type=int, default=5)



    args=parser.parse_args()
    args.video_path = "./RAVDESS_12/"
    args.rgb_max = 1.0
    args.fp16 = False
    
    #args.start = 0#args.frames//2#+1
    #args.alpha = 6/(args.frames-1)
    #args.margin = 3
    args.start = 0

    flow_warping = Resample2d().to(device)
    FlowNet = networks.FlowNet2(args)
    checkpoint = torch.load("weight_files/FlowNet2_checkpoint.pth.tar")
    FlowNet.load_state_dict(checkpoint['state_dict'])
    FlowNet = FlowNet.cuda()
    FlowNet.eval()


    g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=args.resolution))    
    ]))
    g_all.load_state_dict(torch.load(args.weight_file, map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping,g_synthesis=g_all[0],g_all[1]
    perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)

    MSE_Loss=nn.MSELoss(reduction="mean")
    upsample2d=torch.nn.Upsample(scale_factor=256/args.resolution, mode='bilinear') 

    video_list = sorted(os.listdir(args.video_path))
    video_list.reverse()
    for video in video_list:
        param_path = 'inversion_results/ours/'+video+'/codes/'
        inv_path = 'inversion_results/ours/'+video+'/image/'
        make_dir(inv_path)
        make_dir(param_path)

        frame_list = sorted(os.listdir(args.video_path+video+"//"))
        for file_idx in range(0,len(frame_list)):
            if file_idx+args.frames-1 < len(frame_list):
                t_imgfiles = []
                t_names = []

                for index in range(args.frames):
                    t_imgfiles.append(frame_list[file_idx+index])
                for x in t_imgfiles:
                    t_names.append(os.path.basename(x).split(".")[0])
                
                if os.path.isfile(param_path + t_names[args.start] +".npy") == True and os.path.isfile(inv_path + t_names[args.start] +".png") == True:
                    print (video +"/"+ t_names[args.start] +"done!")

                else:

                    Iimgs = []
                    Iimg_ps = []
                    for x in t_imgfiles:
                        Iimg = image_reader(args.video_path+video+'//'+x).to(device)
                        Iimgs.append(Iimg)
                        Iimg_ps.append(upsample2d(Iimg))


                    dlatent=torch.zeros((1,18,512),requires_grad=True,device=device)
                    dlatent.requires_grad=True

                    w_direcs = []
                    for x in range(args.frames-1):
                        w_direc = torch.zeros((1,1,512),requires_grad=True,device=device)
                        w_direc.requires_grad=True
                        w_direcs.append(w_direc)
                    optimizer = optim.Adam(itertools.chain({dlatent},{w_direcs[0]},{w_direcs[1]},{w_direcs[2]},{w_direcs[3]}), lr=0.01,betas=(0.9,0.999),eps=1e-8)

                    flows_forward = []
                    flows_backward = []
                    warps_Iforward = []
                    warps_Ibackward = []

                    for index,x in enumerate(Iimgs):
                        if index != args.start:
                            Fflow = FlowNet(x,Iimgs[args.start])
                            flows_forward.append(Fflow)
                            
                            Fwarp = flow_warping(Iimgs[args.start], Fflow)  
                            warps_Iforward.append(Fwarp.detach())


                            Bflow = FlowNet(Iimgs[args.start],x)
                            flows_backward.append(Bflow)
                
                            Bwarp = flow_warping(x, Bflow)  
                            warps_Ibackward.append(Bwarp.detach())


                    w_pbar = tqdm(range(args.iteration))
                    for i in w_pbar:

                        dlatents = []
                        dlatents.append(dlatent) 
                        dlatents.append(dlatent+w_direcs[0]) 
                        dlatents.append(dlatent+w_direcs[0]+w_direcs[1]) 
                        dlatents.append(dlatent+w_direcs[0]+w_direcs[1]+w_direcs[2]) 
                        dlatents.append(dlatent+w_direcs[0]+w_direcs[1]+w_direcs[2]+w_direcs[3]) 


                        optimizer.zero_grad()

                        Gimgs = []
                        for index,x in enumerate(dlatents):
                            Gimg = (g_synthesis(x) + 1.0) / 2.0
                            Gimgs.append(Gimg)

                        if i == 0:
                            gen_init = Gimgs

                        mse_losses,perceptual_losses = caluclate_loss(torch.stack(Gimgs).squeeze(),torch.stack(Iimgs).squeeze(),perceptual_net,torch.stack(Iimg_ps).squeeze(),MSE_Loss,upsample2d)


                        warps_Gforward = []
                        warps_Gbackward = []
                        for index,x in enumerate(Gimgs):
                            if index != args.start:
                                if index > args.start:
                                    index = index-1
                                    
                                Fwarp = flow_warping(Gimgs[args.start], flows_forward[index])  
                                warps_Gforward.append(Fwarp)
                                Bwarp = flow_warping(x, flows_backward[index])  
                                warps_Gbackward.append(Bwarp)


                        tc_losses = MSE_Loss(torch.stack(warps_Iforward), torch.stack(warps_Gforward)) + MSE_Loss(torch.stack(warps_Ibackward), torch.stack(warps_Gbackward))
                        
                        losses = mse_losses + perceptual_losses + tc_losses #+ distance_losses + distance_loss_ours
                        losses.backward(retain_graph=True)
                        optimizer.step()

                        w_pbar.set_description(
                            (
                                f'loss:  {losses.item():.4f}; perceptual: {perceptual_losses.item():.4f}; mse: {mse_losses.item():.4f}; tc: {tc_losses.item():.4f}'
                            )
                        )
                        del losses,mse_losses,perceptual_losses,tc_losses

                    save_image(Gimgs[args.start].squeeze(0).clamp(0,1),inv_path+"/{}.png".format(t_names[args.start]))
                    np.save(param_path+"{}.npy".format(t_names[args.start]),dlatents[args.start].detach().cpu().numpy())
 







def caluclate_loss(synth_img,img,perceptual_net,img_p,MSE_Loss,upsample2d):
     #calculate MSE Loss
     mse_loss=MSE_Loss(synth_img,img) # (lamda_mse/N)*||G(w)-I||^2

     #calculate Perceptual Loss
     real_0,real_1,real_2,real_3=perceptual_net(img_p)
     synth_p=upsample2d(synth_img) 
     synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)

     perceptual_loss=0
     perceptual_loss+=MSE_Loss(synth_0,real_0)
     perceptual_loss+=MSE_Loss(synth_1,real_1)
     perceptual_loss+=MSE_Loss(synth_2,real_2)
     perceptual_loss+=MSE_Loss(synth_3,real_3)

     return mse_loss,perceptual_loss




if __name__ == "__main__":
    main()
