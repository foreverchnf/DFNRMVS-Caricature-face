import numpy as np
import torch
import torchvision
import trimesh
import matplotlib.pyplot as plt
import os
import cv2

# External libs
import face_alignment
import face_alignment.detection.sfd as face_detector_module

# Internal libs
import data.BFM.utils as bfm_utils
import core_dl.module_util as dl_util
from networks.sub_nets import FNRMVSNet
from demo_utils import *
#from fusing_utils import *

def preprocess(img_dir, fa, face_detector):
    """
    Propare data for inferencing.
    img_dir: directory of input images. str.
    fa: face alignment model. From https://github.com/1adrianb/face-alignment
    face_detector: face detector model. From https://github.com/1adrianb/face-alignment
    """
    transform_func = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img_list = os.listdir(img_dir)

    img_tensors = []
    ori_img_tensors = []
    kpts_tensors = []
    for image_name in img_list:
        image_path = os.path.join(img_dir, image_name)
        img_tensor, ori_img_tensor, kpts_tensor = load_img_2_tensors(image_path, fa, face_detector, transform_func)
        img_tensors.append(img_tensor)
        ori_img_tensors.append(ori_img_tensor)
        kpts_tensors.append(kpts_tensor)
    img_tensors = torch.stack(img_tensors, dim=0).unsqueeze(0)                  # (1, V, C, H, W)
    ori_img_tensors = torch.stack(ori_img_tensors, dim=0).unsqueeze(0)          # (1, V, C, H, W)
    kpts_tensors = torch.stack(kpts_tensors, dim=0).unsqueeze(0)                # (1, V, 68, 2)

    return img_tensors.cuda(), ori_img_tensors.cuda(), kpts_tensors.cuda()


def init_model(checkpoint_path):
    model = FNRMVSNet(opt_step_size=1e-2)
    # pre_checkpoint = dl_util.load_checkpoints(checkpoint_path)
    # model.load_state(pre_checkpoint['net_instance'])
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
    model.eval()
    model.training = False

    bfm = model.opt_layer.bfm
    MM_base_dir = './external/face3d/examples/Data/BFM'
    bfm_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
    face_region_mask = bfm.face_region_mask.copy()
    #face_region_mask[bfm_info['nose_hole'].ravel()] = False
    model.face_region_mask = face_region_mask

    return model


def predict(model, img, ori_img, kpts):
    bfm = model.opt_layer.bfm
    bfm_torch = model.opt_layer.bfm_torch

    # Network forward
    with torch.no_grad():
        sp_norm, ep_norm, pose, colors, vis_mask, full_face_vis_mask, vert_img, ori_colors, \
        opt_verts, opt_verts_img, opt_vis_masks, opt_full_face_vis_masks, ori_colors_list, colors_list, _, _, _ = \
            model.forward(img, ori_img, kpts, None, False)
    opt_verts = correct_landmark_verts(opt_verts, bfm, bfm_torch)

    print(ori_colors_list[-1][-1].shape)
    #print(pose)

    # print(colors_list)
    # print(ori_colors_list)

    # Crop valid face region
    tri = np.zeros_like(bfm.model['tri'])
    tri[:, 0] = bfm.model['tri'][:, 2]
    tri[:, 1] = bfm.model['tri'][:, 1]
    tri[:, 2] = bfm.model['tri'][:, 0]

    N, V, nver, _ = opt_verts[-1][-1].shape

    face_full = []
    face_valid = []
    for i in range(N):
        for j in range(V):
            vert = opt_verts[-1][-1][i, j, :, :].detach().cpu().numpy()
            colors = ori_colors_list[-1][-1][i, :, :].detach().cpu().numpy()
            colors = colors.T
            #print(colors)
            #print(vert)
            #print(vert.shape,vert[0])
            #print(tri.shape,tri[0])
            face_full.append((vert, tri, colors))
            __ = np.ones((vert.shape[0], 3), dtype=np.float)
            vert_valid, _, tri_valid = bfm_utils.filter_non_tight_face_vert(vert, __, tri, model.face_region_mask)
            face_valid.append((vert_valid, tri_valid, colors))
           # print(len(vert_valid))
            #print(vert_valid[0])
            #print(len(face_full),face_full[0])
            


    # Return predicted full mesh (BFM topology) and cropped valid mesh
    # Results are in normalized image space
    # (x-axis to right, y-axis to up, right hand coord, camera center at z-axis facing -z)
    # (can be directly moved to image space by just adding a 2D translation)
    return face_full, face_valid


def visualize(out_dir, img_dir, face_meshes, img, face_region_mask):
    vis_list = []
    V = len(face_meshes)
    for i in range(V):
        vert, tri, colors = face_meshes[i]
        cur_img = img[0, i, ...].cpu()
        cur_img_np = np.ascontiguousarray(cur_img.numpy().transpose((1, 2, 0)))
        viss, colorss = visualize_geometry(colors, out_dir, i, img_dir, vert, np.copy(cur_img_np), tri, face_region_mask, True)
        vis = viss.transpose((1, 2, 0))
        vis_list.append(vis)
    return vis_list, colors

def Fusing(img_dir, caricature, out_dir):
    for filename in os.listdir(img_dir):
        print(filename)
        origin = cv2.imread(os.path.join(img_dir, filename))
        #cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
        # cv2.imshow("origin", origin)
        # cv2.waitKey(0)
        # plt.imshow(origin)
        # plt.show()
        break
    caricature = caricature.astype(origin.dtype)
    #origin = cv2.imread(img_dir)
    mask = 255 * np.ones(caricature.shape, caricature.dtype)
    
    width, height, channels = origin.shape
    center = (height // 2, width // 2)
    print(caricature.shape)
    print(origin.shape)
    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(caricature, origin, mask, center, cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(caricature, origin, mask, center, cv2.MIXED_CLONE)
    mono_clone = cv2.seamlessClone(caricature, origin, mask, center, cv2.MONOCHROME_TRANSFER)
    #out_dir = os.path.join(out_dir, )
    # Write results
    cv2.imwrite(out_dir + "/normal_merge.jpg", normal_clone)
    cv2.imwrite(out_dir + "/fluid_merge.jpg", mixed_clone)
    cv2.imwrite(out_dir + "/mono_merge.jpg", mono_clone)



def save_outputs(out_dir, face_full, face_valid, vis_list, caricature, colors):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    V = len(face_full)
    for i in range(V):
        vert, tri, colors = face_full[i]
        colors = np.minimum(np.maximum(colors, 0), 1)#.transpose((2, 0, 1))
        mesh = trimesh.base.Trimesh(vertex_colors = colors, vertices=vert, faces=tri)
        mesh_path = 'face_full_view%d.obj' % i
        mesh.export(os.path.join(out_dir, mesh_path))

    for i in range(V):
        vert, tri, colors = face_valid[i]
        mesh = trimesh.base.Trimesh(vertex_colors = colors, vertices=vert, faces=tri)
        mesh_path = 'face_valid_view%d.obj' % i
        mesh.export(os.path.join(out_dir, mesh_path))

    for i in range(V):
        vis = vis_list[i]
        vis_path = 'vis_view%d.jpg' % i
        plt.imsave(os.path.join(out_dir, vis_path), vis)



if __name__ == '__main__':
    img_dir = './examples/case_1'
    out_dir = './out_dir/case_19'
    checkpoint_path = './net_weights/2views_model.pth'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=True)
    face_detector = face_detector_module.FaceDetector(device='cuda', verbose=False)

    print('Initializing model ...')
    model = init_model(checkpoint_path)
    print('Preprocessing images ...')
    img, ori_img, kpts = preprocess(img_dir, fa, face_detector)
    print('Reconstructing ...')
    face_full, face_valid = predict(model, img, ori_img, kpts)
    print('Visualizing results ...')
    vis_list, colors = visualize(out_dir, img_dir, face_full, ori_img, model.face_region_mask)

    caricature = Caricatureface(colors, out_dir)
    Fused = Fusing(img_dir, caricature, out_dir)
    # vis_list = visualize(face_valid, img, None)
    print('Saving ...')
    save_outputs(out_dir, face_full, face_valid, vis_list, caricature, colors)
