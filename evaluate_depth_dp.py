from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from layers_dp import disp_to_depth, ScaleRecovery, Project3D, BackprojectDepth, SSIM
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
from math import sqrt
from scipy.stats import norm

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

def compute_grad(color):
    """Computes gradient of an RGB image
    """
    gray = color.convert('L')
    img = np.asarray(gray)
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobelxy

def selected_points(cords, input_img, i):
    """visualize chosen low/high/median texture point
    """
    fig, ax = plt.subplots(1,dpi=300)
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
    ax.scatter(cords[0][1], cords[0][0], s=1, c='g')
    ax.scatter(cords[1][1], cords[1][0], s=1, c='r')
    ax.scatter(cords[2][1], cords[2][0], s=1, c='b')
    plt.savefig(os.path.join(os.path.dirname(__file__), "selected_points","{:06d}.png".format(i)))
    plt.close()

def visual_reprojection(input_img, cords, pixel_cords, i):
    """Visualize reprojected pixel point with the minimum loss
    """
    fig, ax = plt.subplots(1,dpi=300)
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
    ax.scatter(cords[1], cords[0], s=1, c='g')
    ax.scatter(pixel_cords[1], pixel_cords[0], s=1, c='r')
    plt.savefig(os.path.join(os.path.dirname(__file__), "reprojected_points","{:06d}.png".format(i)))
    plt.close()

def create_scatter(ratio_tmp,mask):
    x,y = np.where(mask==1)
    values = ratio_tmp[mask==1]
    return x,y,values


def find_cord(grad,mask):
    """For pixels where LIDAR depth are available, find the low/high/median texture index
    """
    cords = []
    ones = np.ones(grad.shape)
    x_indices,y_indices,values = create_scatter(grad,mask)
    print(values)
    values_tens = torch.from_numpy(values)
    _,min_ind = torch.min(values_tens,0)
    _,max_ind = torch.max(values_tens,0)
    _,median_ind = torch.median(values_tens,0)
    x = x_indices[min_ind]
    y = y_indices[min_ind]
    cords.append([x,y])
    x = x_indices[max_ind]
    y = y_indices[max_ind]
    cords.append([x,y])
    x = x_indices[median_ind]
    y = y_indices[median_ind]
    cords.append([x,y])
    cords_arr = np.array(cords)
    print(cords_arr)

    return cords_arr

def compute_reprojection_loss(pred, target, cords):
    """Computes reprojection loss between a batch of predicted and target images
    """
    ssim = SSIM()

    l1_loss = torch.abs(target - pred).mean(1,True)
    #print("l1_loss mean shape", l1_loss.shape)

    ssim_loss = ssim(pred, target).mean(1,True)
    #print("ssim_loss shape", ssim_loss.shape)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return l1_loss[0,0,cords[0],cords[1]], ssim_loss[0,0,cords[0],cords[1]], reprojection_loss[0,0,cords[0],cords[1]]

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
    x,y,values = create_scatter(ratio_tmp,mask)
    ax.imshow(input_img)
    ax.scatter(y,x,s=0.1,c=values,alpha=0.5,cmap='jet',vmin=-1,vmax=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), "blend_imgs","{:010d}.png".format(i)))
    #ax.imshow(ratio_tmp,cmap='plasma')
    #plt.savefig('tmp.png')
    #img_tmp = pil_loader('tmp.png')
    #print(input_img.size,img_tmp.size)
    #img_blend = Image.blend(input_img, img_tmp,0.5)
    #img_blend.save(os.path.join(os.path.dirname(__file__), name,"{:006d}.png".format(i)))
    plt.close()
    #return img_blend

def tensor_to_PIL(tensor,idx):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    print(image.shape)
    image = unloader(image)
    return image

def get_image_path(folder, frame_index, side):
    #side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    f_str = "{:06d}{}".format(frame_index, '.png')
    image_path = os.path.join(
        '/mnt/sdb/kitti_odometry_dataset',
        "sequences/{:02d}".format(int(folder)),
        "image_2", f_str)
    return image_path

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    selected_frame = 100

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

        sequence_id = 0
        filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIOdomDataset(
            opt.data_path, filenames, opt.height, opt.width,
            [0, 1], 4, is_train=False)
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

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths_odom_00.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    pred_poses = np.load('pred_poses_T.npy')
    norms_divs = np.load('gt_norms_div00.npy')
    scales_dgc = np.load('ratios_of_odom.npy')
    '''
    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))
    '''

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
        if i==selected_frame:
            color_grad = compute_grad(color)     
            color_next = pil_loader(get_image_path(folder,int(frame_index)+1,side))
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
            ratio1,surface_normal1,ground_mask1,_,_,_,_ = scale_recovery(pred_depth)
            ratio = ratio1.cpu().item()
            
            surface_normal = surface_normal1.cpu()[0,:,:,:].numpy()
            ground_mask = ground_mask1.cpu()[0,0,:,:].numpy()
            pred_depth = pred_depth[0].cpu().numpy()
        else:
            ratio = 1
        #print(ratio)
        #print(max(pred_depth))
        #print(min(pred_depth))
        if i==selected_frame:
            cords = find_cord(color_grad, mask)
            selected_points(cords, color, i)
            min_gt = gt_depth[cords[0][0]][cords[0][1]]
            max_gt = gt_depth[cords[1][0]][cords[1][1]]
            median_gt = gt_depth[cords[2][0]][cords[2][1]]
            print("min max median gt depths are", min_gt, max_gt, median_gt)

            to_tensor = transforms.ToTensor()
            color_tens = to_tensor(color)
            color_tens_next = to_tensor(color_next).unsqueeze(0)
            pred_pose = pred_poses[i]
            norms_div = norms_divs[i]
            scale_dgc = scales_dgc[i]
            pred_pose_tens = torch.from_numpy(pred_pose).unsqueeze(0).cuda()
            t_norm = np.linalg.norm(pred_pose[:3, 3])
            print("gt_depth of min max median divided by norms of translation and scale of norm", min_gt/(t_norm*norms_div), max_gt/(t_norm*norms_div), median_gt/(t_norm*norms_div))
            print("gt_depth of min max median divided by norms of translation and scale of dgc", min_gt/(t_norm*scale_dgc), max_gt/(t_norm*scale_dgc), median_gt/(t_norm*scale_dgc))

            depth_tens = torch.from_numpy(pred_depth).unsqueeze(0).cuda()
            project_3d = Project3D(1, gt_height, gt_width).cuda()
            backproject_depth = BackprojectDepth(1, gt_height, gt_width).cuda()
            K_tens = torch.from_numpy(K).unsqueeze(0).cuda()
            inv_K = np.linalg.pinv(K)
            inv_K = torch.from_numpy(inv_K).unsqueeze(0).cuda()
            cam_points = backproject_depth(depth_tens, inv_K,torch.from_numpy(cords[2]).cuda())
            pix_coords = np.array(project_3d(cam_points, K_tens, pred_pose_tens))	
            #print(pix_coords.shape)
            #pix_coords = pix_coords[0,:,:,:]
            l1_losses = []
            ssim_losses = []
            reprojection_losses = []
             
            for pix_coord in pix_coords:
                pix_coord_tens = torch.from_numpy(pix_coord).unsqueeze(0)
                pred = F.grid_sample(color_tens_next, pix_coord_tens, padding_mode="border")
                l1_loss, ssim_loss, reprojection_loss = compute_reprojection_loss(pred, color_tens.unsqueeze(0),cords[2])
                l1_losses.append(l1_loss)
                ssim_losses.append(ssim_loss)
                reprojection_losses.append(reprojection_loss)
            min_loss_pixel_index = np.argmin(reprojection_losses)
            visual_reprojection(color,cords[2],pix_coords[min_loss_pixel_index,cords[2,0],cords[2,1]],selected_frame)

        pred_depth_ori = pred_depth*mask
        gt_depth_ori = gt_depth*mask
        pred_depth_ori = np.where(mask==1,pred_depth_ori,1)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        mean_scale.append(np.mean(gt_depth/pred_depth))
        
        '''
        mu = np.mean(div_values1)
        sigma = np.std(div_values1)
        #print(min(div_values1),max(div_values1))
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
        pred_depth *= ratio
        ratios.append(ratio)

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        #blending_imgs(div_scale, color,i,mask)

        if len(gt_depth) != 0:
            errors.append(compute_errors(gt_depth, pred_depth))
    save_path = os.path.join(os.path.dirname(__file__), "l1_losses_{}.npy".format(selected_frame))
    np.save(save_path, l1_losses)
    save_path = os.path.join(os.path.dirname(__file__), "ssim_losses_{}.npy".format(selected_frame))
    np.save(save_path, ssim_losses)
    save_path = os.path.join(os.path.dirname(__file__), "reprojection_losses_{}.npy".format(selected_frame))
    np.save(save_path, reprojection_losses)
    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
