from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from layers import disp_to_depth, ScaleRecovery
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
from math import sqrt
from scipy.stats import norm

from fit_plane_LSE import fit_plane_LSE_RANSAC

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            #img1 = img.crop((0,191,640,383))
            #return img1
def create_scatter(ratio_tmp,mask):
    
    x,y = np.where(mask==1)
    values = ratio_tmp[mask==1]
    return x,y,values

def create_scatter_p(mask):
    x,y = np.where(mask==1)
    return x,y

def blending_imgs(ratio_tmp, input_img,i,mask):
    """visualizing the factors affecting scale
    """
    fig, ax = plt.subplots(1,dpi=300)
    height, width = ratio_tmp.shape
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width )
    ax.axis('off')
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    #x,y,values = create_scatter(ratio_tmp,mask)
    x,y = create_scatter_p(mask)
    ax.imshow(input_img)
    #ax.scatter(y,x,s=0.1,c=values,alpha=0.5,cmap='jet',vmin=-1,vmax=10)
    ax.scatter(y,x,s=1,c='r',alpha=0.2)
    #plt.savefig(os.path.join(os.path.dirname(__file__), "blend_imgs","{:010d}.png".format(i)))
    plt.savefig(os.path.join(os.path.dirname(__file__), "ground_masks_kitti","{:010d}.png".format(i)))
    #ax.imshow(ratio_tmp,cmap='plasma')
    #plt.savefig('tmp.png')
    #img_tmp = pil_loader('tmp.png')
    #print(input_img.size,img_tmp.size)
    #img_blend = Image.blend(input_img, img_tmp,0.5)
    #img_blend.save(os.path.join(os.path.dirname(__file__), name,"{:006d}.png".format(i)))
    plt.close()
    #return img_blend

def blending_imgs_inliers(u, v, uo, vo, input_img, i):
    """visualizing ground points inliers after RANSAC
    """
    fig, ax = plt.subplots(1,dpi=300)
    #height, width = input_img.size
    width, height = input_img.size
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width )
    ax.axis('off')
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    ax.imshow(input_img)
    ax.scatter(u,v,s=1,c='g',alpha=0.2)
    ax.scatter(uo, vo, s=1, c='r',alpha=0.2)
    plt.savefig(os.path.join(os.path.dirname(__file__), "ground_in_vs_outliers0.4z","{:010d}.png".format(i)))
    plt.close()
 

def tensor_to_PIL(tensor,idx):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    print(image.shape)
    image = unloader(image)
    return image

def get_image_path(folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, '.jpg')
        image_path = os.path.join(
            '/mnt/sdb/xuefeng_data/kitti_data', folder, "image_0{}/data".format(side), f_str)
        return image_path

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    K = np.array([[0.58, 0, 0.5, 0],
                  [0, 1.92, 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(
            opt.data_path, filenames,
            encoder_dict['height'], encoder_dict['width'],
            [0], 4, is_train=False)
        dataloader = DataLoader(
            dataset, 16, shuffle=False, num_workers=opt.num_workers,
            pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.eval_object:
        object_masks = []
        for line in filenames:
            line = line.split()
            folder, frame_index = line[0], int(line[1])

            object_mask_filename = os.path.join(
                os.path.dirname(__file__),
                "object_masks",
                folder,
                "{:010d}.npy".format(int(frame_index)))
            object_mask = np.load(object_mask_filename)
            object_masks.append(object_mask)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
            "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.scaling = "disable" 
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    ex_logs = []
    mean_scale = []
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    #resize_ori = transforms.Resize((pred_disps.shape[1],pred_disps.shape[2]),interpolation=Image.ANTIALIAS)

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        line = filenames[i].split()
        folder = line[0]
        frame_index = line[1]
        side = side_map[line[2]]
        color = pil_loader(get_image_path(folder,int(frame_index),side))
        #color = pil_loader('/mnt/sdb/xuefeng_data/dkit_dataset/20200629_mechanical_fast/images/{:006d}.png'.format(i))
        #color = color.crop((0,191,640,383))
        

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array(
                [0.40810811 * gt_height, 0.99189189 * gt_height,
                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            if opt.eval_object:
                object_mask = object_masks[i].astype(np.bool)

        else:
            mask = gt_depth > 0
        
        if opt.scaling == "gt":
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            if opt.eval_object:
                mask = np.logical_and(mask, object_mask)
        elif opt.scaling == "dgc":
            scale_recovery = ScaleRecovery(1, gt_height, gt_width, K).cuda()
            #scale_recovery = ScaleRecovery(1, 192, 640, K).cuda()
            pred_depth = torch.from_numpy(pred_depth).unsqueeze(0).cuda()
            ratio1,surface_normal1,ground_mask1,cam_points1 = scale_recovery(pred_depth)
            #ratio = ratio1.cpu().item()
            surface_normal = surface_normal1.cpu()[0,0,:,:].numpy()
            ground_mask = ground_mask1.cpu()[0,0,:,:].numpy()
            pred_depth = pred_depth[0].cpu().numpy()
            cam_points=cam_points1.cpu().numpy()
            cam_points2=cam_points.transpose(1,2,0)
            print(np.sum(ground_mask))
            zmax = np.max(cam_points2[:,:,2])
            v,u = np.where((ground_mask==1)&(cam_points2[:,:,2]<0.5*zmax))
            print(v.shape)
            cam_points_masked = cam_points2[np.where((ground_mask==1)&(cam_points2[:,:,2]<0.4*zmax))]
            cam_points_masked_num = np.concatenate((cam_points_masked,np.expand_dims(range(cam_points_masked.shape[0]),1)), axis=1)
            np.random.shuffle(cam_points_masked_num) 
            cam_points4 = np.array(cam_points_masked_num)
            #print(cam_points4.shape)
            cam_points4 = cam_points4[:opt.points_num,:]
            cam_points_selected_index_in_gm = cam_points4[:,-1]
            cam_points3 = np.concatenate((cam_points4[:,:-1], np.ones((cam_points4.shape[0], 1))), axis=1)
            #print(cam_points3.shape)
            #if i==28:
            #    np.savetxt('groundpointsNx428.txt',cam_points4)
            plane,inliers,outliers = fit_plane_LSE_RANSAC(cam_points3,opt.iters,opt.in_t,return_outlier_list=True)
            cam_inliers = cam_points_selected_index_in_gm[inliers]
            cam_outliers = cam_points_selected_index_in_gm[outliers]
            #print(cam_inliers,type(cam_inliers))
            u_inliers = u[cam_inliers.astype(int)]
            v_inliers = v[cam_inliers.astype(int)]
            u_outliers = u[cam_outliers.astype(int)]
            v_outliers = v[cam_outliers.astype(int)]
            blending_imgs_inliers(u_inliers,v_inliers,u_outliers,v_outliers,color,i)
            #print(plane)
            ratio_rans = abs(1.65 / plane[-1])
        else:
            ratio = 1
        #print(ratio)
        #print(max(pred_depth))
        #print(min(pred_depth))
        
        pred_depth_ori = pred_depth*mask
        gt_depth_ori = gt_depth*mask
        pred_depth_ori = np.where(mask==1,pred_depth_ori,1)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        #mean_scale.append(np.mean(gt_depth/pred_depth))

        '''
        error_try = 100
        scale_abs = 0 
        for ratio_try in np.arange(0.1,50,step=0.1):
            pred_depth1=pred_depth * ratio_try
            error_tmp = compute_errors(gt_depth, pred_depth1)[0]
            #print(error_tmp)
            if error_tmp < error_try:
                error_try = error_tmp
                scale_abs = ratio_try
        div_scale = gt_depth_ori / pred_depth_ori
        #print(div_scale.shape)
        div_values1 = div_scale[mask]
        div_scale = (div_scale-scale_abs)/scale_abs
        div_values = div_scale[mask]
        div_rmse = sqrt(sum((div_values1-scale_abs)*(div_values1-scale_abs))/len(div_values1))
        print(min(div_values),max(div_values))
        ex_logs.append([i,min(div_values), max(div_values), div_rmse,scale_abs])
        #print(div_scale.shape)
        #div_scale = div_scale/np.max(div_scale)
        mu = np.mean(div_values1)
        sigma = np.std(div_values1)
        print(min(div_values1),max(div_values1))
        fig,ax=plt.subplots()
        n, bins, patches = ax.hist(div_values1,150,range=(3,130),density = True)
        y = norm.pdf(bins, mu, 0.8*sigma)
        ax.plot(bins, y, 'r')
        plt.xlabel('Scale')
        plt.ylabel('Density')
        plt.savefig(os.path.join(os.path.dirname(__file__), "hist_imgs2","{:010d}.jpg".format(i)))
        plt.close()
        
        #blend_img = blending_imgs(div_scale, color,i)
        #blend_img.save(os.path.join(os.path.dirname(__file__), "blend_imgs","{:010d}.jpg".format(i)))
        
        
        blending_imgs(surface_normal,color,i,'surface_normals')
        blending_imgs(ground_mask,color,i,'ground_masks')
        '''
        #blending_imgs(ground_mask,color,i,ground_mask)
        pred_depth *= ratio_rans
        ratios.append(ratio_rans)

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        #blending_imgs(div_scale, color,i,mask)

        if len(gt_depth) != 0:
            errors.append(compute_errors(gt_depth, pred_depth))
    '''
    fl = open('ex.txt','w')
    fl.writelines(str(ex_logs))
    fl.close()
    '''
    #np.save('mean_scale.npy', mean_scale)

    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    fl = open('{}x{}x{}.txt'.format(opt.points_num,opt.iters,opt.in_t),'w')
    fl.writelines(str(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"))
    fl.close()
 
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
