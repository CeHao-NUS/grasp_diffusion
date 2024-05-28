

python scripts/sample/real_sample_test.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_multi' \
--pc_path '/home/crslab/kelvin/hc_pc/bottle_test_pc.npy' --save_dir '../temp_save/' --show \
 --cond "[0. , 1., 0.]" --inpaint 


scene_bottle_pc
scene_bowl_pc
scene_can_pc
scene_cup_pc
scene_fork_pc
scene_hammer_pc
scene_knife_pc
scene_mug_pc



python vis_single.py