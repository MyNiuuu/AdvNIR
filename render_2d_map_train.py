import blenderproc as bproc
import argparse
import cv2
import os
import numpy as np
import math
import random
from tqdm import tqdm
from PIL import Image

from blenderproc.python.loader.AMASSLoader import _AMASSLoader


# colors = np.random.randint(0, 255, (31 + 1, 3))


def show_fill_map(fillmap):
    """Mark filled areas with colors. It is useful for visualization.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    # Generate color for each fill randomly.
    
    
    
    # colors = np.zeros((np.max(fillmap) + 1, 3))
    # for i in range(np.max(fillmap) + 1):
    #     colors[i] = [255, 255, 255]
    # Id of line is 0, and its color is black.
    # colors[0] = [0, 0, 0]

    colors = np.load('colors.npy')

    # print(colors.shape)
    
    colors = np.concatenate([np.array([[0, 0, 0]]), colors], axis=0) * 255
    

    return colors[fillmap]


if __name__ == '__main__':
    
    OBJ_NUM = 31
    CAMERA_NUM = 25
    RADIUS = [5, 10]
    ANGLE = [-180, 180]
    YAW = [60, 90]

    root = './splited_meshes'
    save_root = './rendered_2d_maps/train'

    # root = '/tempssd/BlenderProc/split_meshes_paper'
    # save_root = '/tempssd/BlenderProc/render_split_meshes_paper'

    lst = [os.path.join(root, x) for x in os.listdir(root)]

    bproc.init()
    bproc.camera.set_resolution(160, 320)

    for name in tqdm(lst):

        bproc.clean_up()

        for i in range(OBJ_NUM):
            objs = bproc.loader.load_obj(os.path.join(name, str(i) + '.obj'))
            objs[0].set_shading_mode("smooth")
            # if i == 19:
            #     _AMASSLoader.correct_materials(objs)

        # define a light and set its location and energy level
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_energy(100)

        # Find point of interest, all cam poses should look towards it
        # poi = bproc.object.compute_poi(objs)
        poi = [0, 0, 0]
        # print(poi)
        # assert False
        # Sample five camera poses
        for i in range(CAMERA_NUM):
            # Sample random camera location around the objects
            # location = bproc.sampler.sphere([0, 0, 0], radius=3, mode="SURFACE")
            # location = np.array([0, -5, 0])

            distance = RADIUS[0] + random.random() * (RADIUS[1] - RADIUS[0])
            angle = random.randint(*ANGLE)
            yaw = random.randint(*YAW)

            # distance, angle, yaw = 10, 0, 90
            
            angle = angle / 360 * 2 * math.pi
            yaw = yaw / 360 * 2 * math.pi

            # 根据三角函数计算摄像机的x,y坐标，z坐标不变
            shadow = distance*math.sin(yaw)
            x = shadow * math.sin(angle)
            y = shadow * math.cos(angle)
            z = distance * math.cos(yaw)
            
            location = np.array([x, -y, z])
            light.set_location(location)
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)

        # activate normal and depth rendering
        # bproc.renderer.enable_normals_output()
        # bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(map_by=["instance"])
        # bproc.renderer.set_output_format(enable_transparency=True)

        # render the whole pipeline
        data = bproc.renderer.render()
        # print(data.keys())
        # print(data['instance_segmaps'][0])
        # cv2.imwrite('/tempssd/BlenderProc/temp.png', show_fill_map(data['instance_segmaps'][0]))
        for i in range(CAMERA_NUM):
            # save_dir = f'{name.replace(root, save_root).split(".obj")[0]}_{i}.png'
            save_dir = f'{name.replace(root, save_root).split(".obj")[0]}_{i}.npy'
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            # print(save_dir)
            # assert False
            # np.save('temp', data['instance_segmaps'][i])
            # cv2.imwrite(f'/tempssd/BlenderProc/output/{i}.png', data['colors'][i][:, :, ::-1])
            np.save(save_dir, data['instance_segmaps'][i])
            cv2.imwrite(save_dir.replace('.npy', '.png'), show_fill_map(data['instance_segmaps'][i]))
            # Image.fromarray(data['colors'][i]).convert('L').save(save_dir.replace('.npy', '.png'))
            # cv2.imwrite(save_dir.replace('.npy', '.png'), data['colors'][i])
            # assert False
        # # write the data to a .hdf5 container
        # bproc.writer.write_hdf5(args.output_dir, data)
