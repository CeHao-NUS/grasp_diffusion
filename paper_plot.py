
import os
import pickle
import json

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from se3dif.visualization import grasp_visualization

import io
from PIL import Image

from vis_param import *


def main(obj_class):


    file_name = os.path.join('./saves/results', obj_class, obj_class + '.pkl')

    with open(file_name, 'rb') as f:
        results = pickle.load(f)

    H = results['H']
    P = results['P']
    mesh = results['mesh']
    trans = results['trans']

    
    # '''
    # flatten H
    reshape_H = H.reshape(-1, 16) 

    # create dict
    d = {}
    for i in range(reshape_H.shape[0]):
        d['grasp_' + str(i)] = reshape_H[i].tolist()

    # save as json
    json_file = os.path.join('./saves/results', obj_class, obj_class + '.json')
    with open(json_file, 'w') as f:
        json.dump(d, f)
    # '''


    # scale in plot
    H[..., :3, -1] *=1/8.
    P *=1/8
    mesh = mesh.apply_scale(1/8)

    # visualize results
    # grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=mesh)

    scene = grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=mesh, show=False)

    # exports to obj
    for idx, geometry in enumerate(scene.geometry.values()):
        # create path 

        os.makedirs(os.path.join('./saves/results', obj_class, 'objs'), exist_ok=True)

        if isinstance(geometry, trimesh.Trimesh):
            if idx == 0 :
                file_name = 'objs/main'
            else:
                file_name = 'objs/grasp_' + str(idx-1)
            geometry.export(os.path.join('./saves/results', obj_class, file_name + '.obj'))
        

    if 'Mug' in obj_class:
        scene.camera_transform = mug
    elif  'Bottle' in obj_class:
        scene.camera_transform = bottle
    elif 'Hammer' in obj_class:
        scene.camera_transform = hammer
    elif 'Fork' in obj_class:
        scene.camera_transform = fork

    # scene.show()



    data = scene.save_image(resolution=(int(1080*1.5),1080))
    image = np.array(Image.open(io.BytesIO(data)))




    plt.imshow(image)

    # remove label number
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # save
    image_path = os.path.join('./saves/results', obj_class, obj_class + '.png')
    plt.savefig(image_path, dpi=300)

    # plt.show()


if __name__ == '__main__':
    obj_name_list = ['Mug', 'Bottle', 'Hammer', 'Fork']
    # obj_name_list = ['Fork']
    subfix = ['', "_cond", '_inpt']

    for obj_name in obj_name_list:
        for s in subfix:
            main(obj_name + s)