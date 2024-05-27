
import numpy as np



# ================== Position Store ==================
cond_inpt = {
    # Mug: grasp handle, grasp rim, grasp bottom, pick the mug
    'Mug': [
               [[0. , -1.2, 0.],  # 0 handle -y
               [0.1 , 0.1 , 0.1], 
               [0. , 0. , 0.]],

               [[0. , 0. , 1.2],  # 1 rim +z
               [0. , 0. , 0.], 
               [0. , 0. , 0.]],

               [[0. , 0. , -2.],  # 2 bottom -z
               [0. , 0. , 0.], 
               [0. , 0. , 0.]],

               [[0. , 0. , 0.],  # 3, random
               [0.2, 0.2, 0.2], 
               [0. , 0. , 0.]],

    ],
# Bottle: grasp left, grasp right, grasp bottom, pick the bottle
    'Bottle':      [
              [[1.2 , 0. , 0.],  # 0  left +x
               [0.1 , 0.1 , 0.1], 
               [0. , 0. , 0.]],

               [[-1.2 , 0. , 0.],  # 1 right -x
               [0.1 , 0.1 , 0.1], 
               [0. , 0. , 0.]],

               [[0. , 0. , -1.2],  # 2 bottom -z
               [0.1 , 0.1 , 0.1], 
               [0. , 0. , 0.]],

               [[0. , 0. , 0.],  # 3 random
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

    ],
# Hammer: grasp handle, grasp head, use the hammer, hand over the hammer
    'Hammer': [
                [[2.0 , 0. , 0.],  # 0 handle +x
               [0.1 , 0.1, 0.1], 
               [0. , 0. , 0.]],

               [[-1.5 , 0. , 0.],  # 1 head -x
               [0.1 , 0.1 , 0.1], 
               [0. , 0. , 0.]],

               [[3. , 0. , 0.],  # 2 right
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

               [[0. , 0. , 0.],  # 3 left
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

    ],
# Fork: grasp handle, grasp head, use the fork, hand over the fork
    'Fork':    [
                [[0. , -1.0 , 0.],  # 0 handle -y
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

               [[0. , +0.8 , 0.],  # 1 head +y
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

               [[0. , -1.2 , 0.],  # 2 use
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

               [[0. , 0., 0.],  # 3 hand
               [0.2 , 0.2 , 0.2], 
               [0. , 0. , 0.]],

    ],
}




chosen_pose = np.eye(4)

def move(bias, rand, radius):
    x0 = np.array([0. , 0. , 0.])
    x0 += np.array(bias)

    x_r = np.random.normal(0. , rand[0])
    y_r = np.random.normal(0. , rand[1])
    z_r = np.random.normal(0. , rand[2])

    # print(x_r, y_r, z_r)

    x0[0] += x_r
    x0[1] += y_r
    x0[2] += z_r

    # print(x0)

    for idx in range(3):
        rad = radius[idx]
        if rad != 0:
            theta = np.random.uniform(0. , 2*np.pi)
            # two other dimensions
            l = [0. ,1,2]
            l.remove(idx)
            x0[l[0]] += rad * np.sin(theta)
            x0[l[1]] += rad * np.cos(theta)

    # print(x0)
    return x0

def translate(x0):
    base = np.eye(4)
    base[:3,-1] = x0
    return base


def init_pose(obs_class, idx=0):
    cond = cond_inpt
    if obs_class in cond.keys():
        bias = cond[obs_class][idx][0]
        rand = cond[obs_class][idx][1]
        radius = cond[obs_class][idx][2]
        x0 = move(bias, rand, radius)

        global chosen_pose
        chosen_pose = translate(x0)

    return chosen_pose

def set_chosen_pose(x0): #[0. , 0. , 0.]
    global chosen_pose
    chosen_pose = translate(x0)