import numpy as np
import torch
from torchvision.utils import make_grid
import os
from PIL import Image
import cv2

# External libs

from external.face3d.face3d import mesh
from external.face3d.face3d.morphable_model.load import load_BFM_info
from external.face3d.face3d.morphable_model import MorphabelModel

import external.face3d.face3d as face3d
import sys
import dlib
import glob
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')


# Internal libs
import data.BFM.utils as bfm_utils

def Caricatureface(colors, out_dir):
    bfm = MorphabelModel('external/face3d/examples/Data/BFM/Out/BFM.mat')
    print('init bfm model success')

    ii = 0
    defo_folder_path = 'deformation'
    obj_name = '/defor_D_1.obj'

    print(1)
    objFilePath = defo_folder_path + obj_name
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
        
    # name = f.replace('deformation','')
    # name = name.replace('\','')
    vertices = np.array(points)

    triangles = bfm.triangles


    # ------------------------------ 2. modify vertices(transformation. change position of obj)
    # -- change the position of mesh object in world space
    # scale. target size=180 for example
    s = 480/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
    # rotate 30 degree for example
    R = mesh.transform.angle2matrix([0, 0, 180]) 
    # no translation. center of obj:[0,0]
    t = [250, 250, 0]
    transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

    # ------------------------------ 3. modify colors/texture(add light)
    # -- add point lights. light positions are defined in world space
    # set lights
    # light_positions = np.array([[-128, -128, 300]])
    # light_intensities = np.array([[1, 1, 1]])
    # lit_colors = mesh.light.add_light(transformed_vertices, triangles, colors, light_positions, light_intensities)

    # ------------------------------ 4. modify vertices(projection. change position of camera)
    # -- transform object from world space to camera space(what the world is in the eye of observer). 
    # -- omit if using standard camera
    camera_vertices = mesh.transform.lookat_camera(transformed_vertices, eye = [0, 0, 100], at = np.array([0, 0, 0]), up = None)
    # -- project object from 3d world space into 2d image plane. orthographic or perspective projection
    projected_vertices = mesh.transform.orthographic_project(camera_vertices)

    # ------------------------------ 5. render(to 2d image)
    # set h, w of rendering
    h = w = 700
    # change to image coords for rendering
    z = vertices[:,2:]
    z = 0 - z
    vertices[:,2:] = z
    y = vertices[:,1:2]
    y = w - y
    vertices[:,1:2] = y
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)

    # z = image_vertices[:,2:]
    # z = 0 - z
    # image_vertices[:,2:] = z
    # y = image_vertices[:,1:2]
    # y = w - y
    # image_vertices[:,1:2] = y
    
    # render 
    rendering =  mesh.render.render_colors(image_vertices, triangles, colors, h, w)
    # ---- show rendering
    # plt.imshow(rendering)
    # plt.show()
    save_folder = out_dir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    io.imsave('{}/Caricature.jpg'.format(save_folder), rendering)
    mesh.io.write_obj_with_colors('{}/defo_face_{}'.format(save_folder,ii), vertices, bfm.triangles, colors)
    return rendering


def fetch_color(img_dir):
    # load BFM model
    global colorss
    bfm = MorphabelModel('external/face3d/examples/Data/BFM/Out/BFM.mat')
    print('init bfm model success')

    #uv_coords = face3d.morphable_model.load.load_uv_coords('examples/Data/BFM/Out/BFM_UV.mat') 
    t = [0, 0, 0]
    s = 8e-03
    c = 3

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    faces_folder_path = img_dir
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    iii = 0
    tp = bfm.get_tex_para('random')
    colorss = bfm.generate_colors(tp)
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        if iii != 0:
            break
        iii += 1
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            centroid_x = (d.left() + d.right())/2
            centroid_y = (d.top() + d.bottom())/2
            h = int(0-(d.top() - d.bottom())*1.2)
            w = int(0-(d.left() - d.right())*1.2)

            x = []
            for pt in shape.parts():
                a = float(pt.x) - centroid_x
                b = float(pt.y) - centroid_y
                tmp = np.array([a, b])
                x.append(tmp)
            x = np.array(x)

            X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.

            # fit
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)

            
            colorss = np.minimum(np.maximum(colorss, 0), 1)

            fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
            transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
            image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

            #Invert y and z axis to make rendering image normal
            # z = image_vertices[:,2:]
            # z = 0 - z
            # image_vertices[:,2:] = z
            # y = image_vertices[:,1:2]
            # y = w - y
            # image_vertices[:,1:2] = y
            fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colorss, h, w)
            color = fitted_image
            
            ## ----get color from original image------------------------------------------------------------
            for xx in range(int(centroid_y)-int(h/2),int(centroid_y)+int(h/2)):
                if xx < 0 or xx >= img.shape[0]:
                    continue
                xxx = int(xx)+int(h/2)-int(centroid_y)
                for yy in range(int(centroid_x)-int(w/2),int(centroid_x)+int(w/2)):
                    if yy < 0 or yy >=img.shape[1]:
                        continue
                    yyy = int(yy)+int(w/2)-int(centroid_x)
                    if fitted_image[xxx, yyy][0] != 0 and fitted_image[xxx, yyy][1] != 0 and fitted_image[xxx, yyy][2] != 0:
                        color[xxx,yyy] = img[xx,yy]
                        #img[xx,yy] = (255*fitted_image[xxx, yyy]).astype(np.uint8)
            
            ## ----add colors to vertices----------------------------------------------------------------
            vvv=0
            for ver in image_vertices:
                if int(ver[1])<fitted_image.shape[0] and int(ver[0])<fitted_image.shape[1]:
                    colorss[vvv] = color[int(ver[1]),int(ver[0])]
                vvv+=1 
    return colorss

def fetch_color_2(img_dir):
    # load BFM model
    global colorss
    bfm = MorphabelModel('external/face3d/examples/Data/BFM/Out/BFM.mat')
    print('init bfm model success')

    t = [0, 0, 0]
    s = 8e-03
    c = 3

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    faces_folder_path = img_dir
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    iii = 0
    tp = bfm.get_tex_para('random')
    colorss = bfm.generate_colors(tp)
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        if iii != 0:
            break
        iii += 1
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            centroid_x = (d.left() + d.right())/2
            centroid_y = (d.top() + d.bottom())/2
            h = int(0-(d.top() - d.bottom())*1.6)
            w = int(0-(d.left() - d.right())*1.6)

            x = []
            for pt in shape.parts():
                a = float(pt.x) - centroid_x
                b = float(pt.y) - centroid_y
                tmp = np.array([a, b])
                x.append(tmp)
            x = np.array(x)

            X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.

            # fit
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)

            
            colorss = np.minimum(np.maximum(colorss, 0), 1)

            fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
            transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
            image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

            #Invert y and z axis to make rendering image normal
            z = image_vertices[:,2:]
            z = 0 - z
            image_vertices[:,2:] = z
            y = image_vertices[:,1:2]
            y = w - y
            image_vertices[:,1:2] = y
            fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colorss, h, w)
            color = fitted_image
            
            ## ----get color from original image------------------------------------------------------------
            for xx in range(int(centroid_y)-int(h/2),int(centroid_y)+int(h/2)):
                if xx < 0 or xx >= img.shape[0]:
                    continue
                xxx = int(xx)+int(h/2)-int(centroid_y)
                for yy in range(int(centroid_x)-int(w/2),int(centroid_x)+int(w/2)):
                    if yy < 0 or yy >=img.shape[1]:
                        continue
                    yyy = int(yy)+int(w/2)-int(centroid_x)
                    if fitted_image[xxx, yyy][0] != 0 and fitted_image[xxx, yyy][1] != 0 and fitted_image[xxx, yyy][2] != 0:
                        color[xxx,yyy] = img[xx,yy]
                        #img[xx,yy] = (255*fitted_image[xxx, yyy]).astype(np.uint8)
            
            ## ----add colors to vertices----------------------------------------------------------------
            vvv=0
            for ver in image_vertices:
                if int(ver[1])<fitted_image.shape[0] and int(ver[0])<fitted_image.shape[1]:
                    colorss[vvv] = color[int(ver[1]),int(ver[0])]
                vvv+=1 
    return colorss


def visualize_geometry(colors_1, out_dir, i, img_dir, vert, back_ground, tri, face_region_mask=None, gt_flag=False):
    """
    Visualize untextured mesh
    :param vert: mesh vertices. np.array: (nver, 3)
    :param back_ground: back ground image. np.array: (256, 256, 3)
    :param tri: mesh triangles. np.array: (ntri, 3) int32
    :param face_region_mask: mask for valid vertices. np.array: (nver, 1) bool
    :param gt_flag: Whether render with ESRC ground truth mesh. The normals of BFM (predicted mesh) point to the
                    opposite direction, thus need to multiply by -1.
    :return: image_t: rendered image. np.array: (3, 256, 256)
    """
    #global colorss
    colors = np.ones((vert.shape[0], 3), dtype=np.float) - 0.25
    if gt_flag:
        sh_coeff = np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    else:
        sh_coeff = np.array((0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    colors = mesh.light.add_light_sh(vert, tri, colors, sh_coeff)
    
    projected_vertices = vert.copy()  # using stantard camera & orth projection
    
    h = w = 256
    c = 3
    image_vert = mesh.transform.to_image(projected_vertices, h, w)

    #print(image_vert.shape, image_vert[0:200])

    # if face_region_mask is not None:
    #     image_vert, colors, tri = bfm_utils.filter_non_tight_face_vert(image_vert, colors, tri, face_region_mask)
    # colorss = fetch_color_2(img_dir)
    # colors = colorss.astype(np.uint8)
    colors = colors/np.max(colors)
    #colors = colors.astype(np.uint8)
    z = projected_vertices[:,2:]
    z = 0 - z
    projected_vertices[:,2:] = z
    
    image_t = mesh.render.render_colors(image_vert, tri, colors_1, h, w, BG=back_ground)

    ## ----make a obj file of the face model(optional)--------------------------------------------
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mesh.io.write_obj_with_colors('{}/face_{}'.format(out_dir, i), projected_vertices, tri, colors_1)

    image_t = np.minimum(np.maximum(image_t, 0), 1).transpose((2, 0, 1))
    return image_t, colors

def visualize_geometry_2(colors_1, out_dir, i, img_dir, vert, back_ground, tri, face_region_mask=None, gt_flag=False):
    """
    Visualize untextured mesh
    :param vert: mesh vertices. np.array: (nver, 3)
    :param back_ground: back ground image. np.array: (256, 256, 3)
    :param tri: mesh triangles. np.array: (ntri, 3) int32
    :param face_region_mask: mask for valid vertices. np.array: (nver, 1) bool
    :param gt_flag: Whether render with ESRC ground truth mesh. The normals of BFM (predicted mesh) point to the
                    opposite direction, thus need to multiply by -1.
    """
    #global colorss
    colors = np.ones((vert.shape[0], 3), dtype=np.float) - 0.25
    if gt_flag:
        sh_coeff = np.array((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    else:
        sh_coeff = np.array((0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float).reshape((9, 1))
    #colors = mesh.light.add_light_sh(vert, tri, colors, sh_coeff)
    
    vertices = vert.copy()  # using stantard camera & orth projection
    s = 680/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
    print("s: ", s)
    # rotate 30 degree for example
    R = mesh.transform.angle2matrix([0, 0, 0]) 
    # no translation. center of obj:[0,0]
    t = [0, 0, 0]
    projected_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

    
    global h, w
    h = w = 256
    c = 3
    #image_vert = mesh.transform.to_image(projected_vertices, h, w)
    
    # if face_region_mask is not None:
    #     image_vert, colors, tri = bfm_utils.filter_non_tight_face_vert(image_vert, colors, tri, face_region_mask)
    # # colorss = fetch_color(img_dir)
    # # colors = colorss.astype(np.uint8)

    bfm = MorphabelModel('external/face3d/examples/Data/BFM/Out/BFM.mat')
    print('init bfm model success')

    #global colorss
    faces_folder_path = img_dir
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    iii = 0
    tp = bfm.get_tex_para('random')
    global image_vert
    image_vert = mesh.transform.to_image(projected_vertices, h, w)
    colors = bfm.generate_colors(tp)
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        if iii != 0:
            break
        iii += 1
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            centroid_x = (d.left() + d.right())/2
            centroid_y = (d.top() + d.bottom())/2
            h = int(0-(d.top() - d.bottom())*1.6)
            w = int(0-(d.left() - d.right())*1.6)

            x = []
            for pt in shape.parts():
                a = float(pt.x) - centroid_x
                b = float(pt.y) - centroid_y
                tmp = np.array([a, b])
                x.append(tmp)
            x = np.array(x)

            #print(h,w)
            #print(img.shape)
            X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.
             
            # fit
            fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)
            #print("fitted: ", fitted_s, fitted_angles, fitted_t)
            angles = [0, 0, 0]
            projected_vertices = bfm.transform(vertices, s, angles, -fitted_t/(2*s))
            image_vert = mesh.transform.to_image(projected_vertices, h, w)

            #Invert y and z axis to make rendering image normal
            z = image_vert[:,2:]
            z = 0 - z
            image_vert[:,2:] = z
            x = image_vert[:,0:1]
            x = h - x
            image_vert[:,0:1] = x

            fitted_image = mesh.render.render_colors(image_vert, bfm.triangles, colors, h, w)
            color = fitted_image
            ## ----get color from original image------------------------------------------------------------
            for xx in range(int(centroid_y)-int(h/2),int(centroid_y)+int(h/2)):
                if xx < 0 or xx >= img.shape[0]:
                    continue
                xxx = int(xx)+int(h/2)-int(centroid_y)
                for yy in range(int(centroid_x)-int(w/2),int(centroid_x)+int(w/2)):
                    if yy < 0 or yy >=img.shape[1]:
                        continue
                    yyy = int(yy)+int(w/2)-int(centroid_x)
                    # if (xxx < 0 or xxx >= h) or (yyy < 0 or yyy >= w):
                    #     continue
                    if fitted_image[xxx, yyy][0] != 0 and fitted_image[xxx, yyy][1] != 0 and fitted_image[xxx, yyy][2] != 0:
                        color[xxx,yyy] = img[xx,yy]
                        #img[xx,yy] = (255*fitted_image[xxx, yyy]).astype(np.uint8)
            
            ## ----add colors to vertices----------------------------------------------------------------
            # vvv=0
            # for ver in image_vert:
            #     if int(ver[1])<fitted_image.shape[0] and int(ver[0])<fitted_image.shape[1]:
            #         #colors[vvv] = color[int(ver[1]),int(ver[0])]
            #     vvv+=1      

    colors = colors/np.max(colors)
    print("h,w:", h, w)
    image_t = mesh.render.render_colors(image_vert, tri, colors_1, 256, 256, BG=back_ground)
    ## ----make a obj file of the face model(optional)--------------------------------------------
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mesh.io.write_obj_with_colors('{}/face_{}'.format(out_dir, i), image_vert, tri, colors_1)

    
    image_t = np.minimum(np.maximum(image_t, 0), 1).transpose((2, 0, 1))
    return image_t, colors

def visualization(verts, img, bfm, n_sample=None, MM_base_dir='./external/face3d/examples/Data/BFM'):
    bfm_info = load_BFM_info(os.path.join(MM_base_dir, 'Out/BFM_info.mat'))
    face_region_mask = bfm.face_region_mask.copy()
    face_region_mask[bfm_info['nose_hole'].ravel()] = False
    N, V, _, _, _ = img.shape
    img_grids = []
    for i in range(N):
        if n_sample is not None and i >= n_sample:
            break
        img_list = []
        for j in range(V):
            cur_img = img[i, j, ...].cpu()
            cur_img_np = np.ascontiguousarray(cur_img.numpy().transpose((1, 2, 0)))
            img_list.append(cur_img)
            for k in range(len(verts)):
                if k == 0:
                    vert = verts[k][i, j, ...].detach().cpu().numpy()
                else:
                    vert = verts[k][-1][i, j, ...].detach().cpu().numpy()

                geo_vis = visualize_geometry(vert, np.copy(cur_img_np), bfm.model['tri'], face_region_mask)
                img_list.append(torch.tensor(geo_vis))
        img_grid = make_grid(img_list, nrow=1 + len(verts)).detach().cpu()
        img_grids.append(img_grid)

    return img_grids


def correct_landmark_verts(verts, bfm, bfm_torch):
    N, V, nver, _ = verts[0].shape

    # Get landmark and neighbor idx
    kpt_neib_idx = bfm.neib_vert_idx[bfm.kpt_ind, :]            # (68, max_number_neighbor_per_vert)
    kpt_neib_idx = kpt_neib_idx[kpt_neib_idx < nver]
    # kpt_idx = np.concatenate([bfm.kpt_ind, kpt_neib_idx], axis=0)

    for k in range(1, len(verts)):
        vert = verts[k][-1]

        # Compute laplacian mean filtered vertices
        vert = vert.view(N * V, nver, 3)
        vert_t = torch.cat([vert, torch.zeros_like(vert[:, :1, :])], dim=1)  # (N * V, nver + 1, 3)
        vert_neib = vert_t[:, bfm.neib_vert_idx.ravel(), :].view(N * V, nver, bfm.neib_vert_idx.shape[1], 3)
        vert_neib_sum = torch.sum(vert_neib, dim=2)  # (N * V, nver, 3)
        vert_lapla_mean = vert_neib_sum / bfm_torch.neib_vert_count.view(1, nver, 1).float()

        # Replace lamdmark vertices with laplacian mean
        vert[:, bfm.kpt_ind, :] = 0.9 * vert_lapla_mean[:, bfm.kpt_ind, :] + 0.1 * vert[:, bfm.kpt_ind, :]

        # # Compute laplacian mean filtered vertices
        # vert = vert.view(N * V, nver, 3)
        # vert_t = torch.cat([vert, torch.zeros_like(vert[:, :1, :])], dim=1)  # (N * V, nver + 1, 3)
        # vert_neib = vert_t[:, bfm.neib_vert_idx.ravel(), :].view(N * V, nver, bfm.neib_vert_idx.shape[1], 3)
        # vert_neib_sum = torch.sum(vert_neib, dim=2)  # (N * V, nver, 3)
        # vert_lapla_mean = vert_neib_sum / bfm_torch.neib_vert_count.view(1, nver, 1).float()
        #
        # # Replace lamdmark vertices with laplacian mean
        # vert[:, kpt_neib_idx, :] = vert_lapla_mean[:, kpt_neib_idx, :]

        verts[k][-1] = vert.view(N, V, nver, 3)

    return verts


def load_img_2_tensors(image_path, fa, face_detector, transform_func=None):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(
        img,
        top=50,
        bottom=50,
        left=50,
        right=50,
        borderType=cv2.BORDER_DEFAULT
    )
    s = 1.5e3
    t = [0, 0, 0]
    scale = 1.2
    size = 256
    ds = face_detector.detect_from_image(img[..., ::-1].copy())
    for i in range(len(ds)):
        d = ds[i]
        center = [d[3] - (d[3] - d[1]) / 2.0, d[2] - (d[2] - d[0]) / 2.0]
        center[0] += (d[3] - d[1]) * 0.06
        center[0] = int(center[0])
        center[1] = int(center[1])
        l = max(d[2] - d[0], d[3] - d[1]) * scale
        if l < 200:
            continue
        x_s = center[1] - int(l / 2)
        y_s = center[0] - int(l / 2)
        x_e = center[1] + int(l / 2)
        y_e = center[0] + int(l / 2)
        t = [256. - center[1] + t[0], center[0] - 256. + t[1], 0]
        rescale = size / (x_e - x_s)
        s *= rescale
        t = [t[0] * rescale, t[1] * rescale, 0.]
        img = Image.fromarray(img).crop((x_s, y_s, x_e, y_e))
        img = cv2.resize(np.asarray(img), (size, size)).astype(np.float32)
        break
    assert img.shape[0] == img.shape[1] == 256
    ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0)  # (C, H, W)
    img_tensor = ori_img_tensor.clone()
    if transform_func:
        img_tensor = transform_func(img_tensor)

    # Get 2D landmarks on image
    kpts_list = fa.get_landmarks(img)
    kpts = kpts_list[0]
    kpts_tensor = torch.from_numpy(kpts)                                                    # (68, 2)

    return img_tensor, ori_img_tensor, kpts_tensor
