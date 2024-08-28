
# 1. auto inpainting option
# 2. save to numpy file, create dir
# 3. save images
# 4. auto change name of file


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--obj_id', type=int, default=0)
    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--obj_class', type=str, default='Laptop')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--eval_sim', type=bool, default=False)
    p.add_argument('--model', type=str, default='grasp_dif_multi')
    p.add_argument('--pc_path', type=str, default='')
    p.add_argument('--save_dir', type=str, default='')
    p.add_argument('--cond', type=str, default='')
    p.add_argument('--inpaint', action='store_true')
    p.add_argument('--show', action='store_true')

    p.add_argument('--method', choices=['vanilla', 'opt', 'goal'], default='opt')

    opt = p.parse_args()
    return opt



def sample_pointcloud(args, obj_id=0, obj_class='Mug'):


    file_path = args.pc_path
    P = np.load(file_path, allow_pickle=True)

    # P = P['P']
    # pc = P
    # cutoff_height = 0.08
    # x, y, z = np.split(pc, 3, axis=-1)
    # mask = (0.40 <= x)*(x <= 0.55)
    # pc = pc[mask[..., 0]]

    # x, y, z = np.split(pc, 3, axis=-1)
    # mask2 = (cutoff_height <= z)*(z <= 0.350)
    # pc = pc[mask2[..., 0]]

    # x, y, z = np.split(pc, 3, axis=-1)
    # mask3 = (-0.02 <= y)*(y <= 0.10)
    # pc = pc[mask3[..., 0]]

    # P = pc
# 
    random_indices = np.random.choice(P.shape[0], 5000, replace=False)
    P = P[random_indices, :3]

    sampled_rot = scipy.spatial.transform.Rotation.random()

    # use fixed rotation for reproducibility
    rot = np.eye(3)
    rot_quat = sampled_rot.as_quat()

    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    H[:3,-1] = -P_mean

    translational_shift = copy.deepcopy(H)

    return P, translational_shift, rot_quat



if __name__ == '__main__':
    import copy
    import configargparse
    args = parse_args()

    EVAL_SIMULATION = args.eval_sim
    # isaac gym has to be imported here as it is supposed to be imported before torch
    if (EVAL_SIMULATION):
        # Alternatively: Evaluate Grasps in Simulation:
        from isaac_evaluation.grasp_quality_evaluation import GraspSuccessEvaluator

    import scipy.spatial.transform
    import numpy as np
    from se3dif.datasets import AcronymGraspsDirectory
    from se3dif.models.loader import load_model
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    from se3dif.utils import to_numpy, to_torch

    import torch



    print('##########################################################')
    # print('Object Class: {}'.format(args.obj_class))
    # print(args.obj_id)
    print('file', args.pc_path)
    print('##########################################################')

    n_grasps = int(args.n_grasps)
    obj_id = int(args.obj_id)
    obj_class = args.obj_class
    n_envs = 30
    device = args.device

    ## Set Model and Sample Generator ##
    P, trans, rot_quad = sample_pointcloud(args, obj_id, obj_class)


    ## Visualize results ##
    from se3dif.visualization import grasp_visualization


    P *=1/8
    scene = grasp_visualization.vis_point_cloud(P, show=args.show)
    

    # ================= save all
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    pc_path = args.pc_path.split('/')[-1]
    pc_path = pc_path.split('.')[0]

    # import pickle
    # file_name = 'save_all_' + pc_path + '.npy'
    # save_dir = os.path.join(args.save_dir, file_name)
    # results = {'H': to_numpy(H), 'P': P, 'trans': trans}
    # with open(save_dir, 'wb') as f:
    #     pickle.dump(results, f)


    
    # ============ save images
    from PIL import Image
    import io
    import matplotlib.pyplot as plt

    data = scene.save_image(resolution=(int(1080*1.5),1080))
    image = np.array(Image.open(io.BytesIO(data)))

    plt.imshow(image)

    # remove label number
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # save
    file_name = 'save_img_' + pc_path + '.png'
    save_dir = os.path.join(args.save_dir, file_name)
    plt.savefig(save_dir, dpi=300)

    plt.close()

