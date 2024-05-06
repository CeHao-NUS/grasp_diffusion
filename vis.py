from se3dif.visualization import grasp_visualization

import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# load pickle file
import pickle



trans = np.array([
       [ 0.        ,  0.        ,  -1.        ,  -3.31721040e-01],
       [ -1.        ,  0.        ,  0.        , -1.80438071e-02],
       [-0.        ,  1.        ,  0        ,  5.92240600e-02],
       [ 0.        ,  1.        ,  0.        ,  1.        ]]) # Mug, 1


if __name__ == '__main__':
    file_hammer = 'results(hammer2).pkl'

    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)


    H = results['H']
    P = results['P']
    mesh = results['mesh']
    trans = results['trans']

    # scale in plot
    H[..., :3, -1] *=1/8.
    P *=1/8
    mesh = mesh.apply_scale(1/8)

    # visualize results
    # grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=mesh)

    '''
    visualize_grasps(Hs, scale=1., p_cloud=None, energies=None, colors=None, mesh=None, show=True):

    '''




    # H0 = H[1]
    # H0 = H[0]
    # H = np.array([H0])

    # H0[0:3,3] = 0
    # H0 = np.eye(4)

    H0 = np.eye(4)
    H1 = np.eye(4)
    # H1[:3,:3] = np.array([
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [0, 1 ,0]
    # ])

    # H1[1, -1]   = - 0.1
    H1[2, -1]   = 0.1

    H = np.array([H0, H1])

    # print(H0)
    # print(H1)


    scene = grasp_visualization.visualize_grasps(H, p_cloud=P, mesh=mesh, show=False)

    scene.camera_transform = trans

    scene.show()


    a = 1


# data = scene.save_image(resolution=(1080,1080))
# image = np.array(Image.open(io.BytesIO(data)))


# image = grasp_visualization.get_scene_grasps_image(H, p_cloud=P, mesh=mesh)

# plt.imshow(image)
# plt.show()