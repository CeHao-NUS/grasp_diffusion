

task_list=('scene_bottle_pc' 'scene_bowl_pc' 'scene_can_pc' 'scene_cup_pc' 'scene_fork_pc' 'scene_hammer_pc' 'scene_knife_pc' 'scene_mug_pc')


# no cond


for task in ${task_list[@]} 
do
    python scripts/sample/real_sample_test.py --n_grasps 40 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_multi' \
    --pc_path '../scene_data/'$task'.npy' --save_dir '../temp_save/' --show
done
    