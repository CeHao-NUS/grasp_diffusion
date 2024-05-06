from se3dif.visualization import grasp_visualization

import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load pickle file
import pickle


if __name__ == '__main__':

    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)


    H = results['H']
    P = results['P']
    mesh = results['mesh']
    trans = results['trans']


    H[..., :3, -1] *=1/8.
    P *=1/8
    mesh = mesh.apply_scale(1/8)

    # image = grasp_visualization.get_scene_grasps_image(H, p_cloud=P, mesh=mesh)

    # plt.imshow(image)
    # plt.show()

    scene = grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=mesh, show=False)

    from vis import trans

    scene.camera_transform = trans

    scene.show()

    data = scene.save_image(resolution=(int(1080*1.5),1080))
    image = np.array(Image.open(io.BytesIO(data)))




    plt.imshow(image)

    # remove label number
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # save
    plt.savefig('grasp.png', dpi=300)

    plt.show()