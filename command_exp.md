

Mug: grasp handle, grasp rim, grasp bottom, pick the mug
Bottle: grasp left, grasp right, grasp bottom, pick the bottle
Hammer: grasp handle, grasp head, use the hammer, hand over the hammer
Fork: grasp handle, grasp head, use the fork, hand over the fork


# =================## mug

## uncon
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_mugs'

## vanilla


## inpainting
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 4 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_mugs' \
--idx 1

## goal
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_mug_cond' \
--idx 0


# =================##  bottle

## uncon
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 4 --obj_id 3 --obj_class 'Bottle' --model 'grasp_bottle' \
--idx 0

## vanilla

## inpainting

## goal
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 4 --obj_id 3 --obj_class 'Bottle' --model 'grasp_bottle_cond' \
--idx 0




# =================##  hammer

## uncon
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 20 --obj_id 1 --obj_class 'Hammer' --model 'grasp_hammer'
[] This only grasp handle

python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 4 --obj_id 1 --obj_class 'Hammer' --model 'grasp_dif_multi' \
--idx 1

## vanilla

## inpainting

## goal
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 4 --obj_id 1 --obj_class 'Hammer' --model 'grasp_hammer_cond' \
--idx 0





# =================##  fork

## uncon
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 4 --obj_id 5 --obj_class 'Fork' --model 'grasp_fork' \
--idx 1

python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 5 --obj_class 'Fork' --model 'grasp_dif_multi'

## vanilla

## inpainting

## goal

python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 5 --obj_id 5 --obj_class 'Fork' --model 'grasp_fork_cond' \
--idx 0