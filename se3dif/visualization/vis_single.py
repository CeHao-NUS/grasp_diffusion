from se3dif.visualization import grasp_visualization

import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# load pickle file
import pickle


def show_image(H, P, save_dir):
    scene = grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=None, show=False)

    # scene.show()

    data = scene.save_image(resolution=(int(1080*1.5),1080))
    image = np.array(Image.open(io.BytesIO(data)))

    plt.imshow(image)

    # remove label number
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # save
    plt.savefig(save_dir, dpi=300)

    # plt.show()
    plt.close()

def vis_every_grasp(file_name, image_dir):

    with open(file_name, 'rb') as f:
        results = pickle.load(f)
    H = results['H']
    P = results['P']
    mesh = results['mesh']
    trans = results['trans']

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i in range(len(H)):
        hi = H[i]
        hi = hi[np.newaxis, ...]
        file_name = 'grasp_' + str(i) + '.png'
        save_dir = os.path.join(image_dir, file_name)
        show_image(hi, P, save_dir)
        
    