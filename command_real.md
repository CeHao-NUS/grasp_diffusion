

python scripts/sample/real_sample_test.py --n_grasps 1 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_multi' \
--pc_path '../scene_data/scene_mug_pc.npy' --save_dir '../temp_save/' --show \
 --cond "[0. , 0., 1.]" --inpaint 

python scripts/sample/vis_real_point_cloud.py --obj_id 10 --obj_class 'Mug' --model 'grasp_dif_multi' \
--pc_path '../scene_data/scene_mug_pc.npy' --save_dir '../temp_save/' --show 

scene_everything_pc
save_all_scene_everything_pc
save_grasp_scene_everything_pc

scene_bottle_pc
scene_bowl_pc
scene_can_pc
scene_cup_pc
scene_fork_pc
scene_hammer_pc
scene_knife_pc
scene_mug_pc



python scripts/sample/real_sample_test.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_multi' --pc_path '/home/crslab/kelvin/hc_pc/test_fork_handle_pc.npy' --save_dir '/home/crslab/kelvin/hc_pc/' --show  --cond "[0 , 1, 0.4]" --inpaint 


python scripts/sample/real_sample_test.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_mugs' --pc_path '/home/crslab/kelvin/hc_pc/test_bottle_pc.npy' --save_dir '/home/crslab/kelvin/hc_pc/' --show  --cond "[0 , 0, 1]" --inpaint 


# example

# =============== for inpainting
## inpainting opt is our method

python scripts/sample/real_sample_test.py --n_grasps 10  --model 'grasp_dif_multi' \
--pc_path '../scene_data/scene_hammer_pc.npy' --save_dir '../temp_save/' --show --method 'opt'\
 --cond "[0. , 1., 2.]" --inpaint 


## inpainting vanilla is our baseline

python scripts/sample/real_sample_test.py --n_grasps 10  --model 'grasp_dif_multi' \
--pc_path '../scene_data/scene_hammer_pc.npy' --save_dir '../temp_save/' --show --method 'vanilla'\
 --cond "[0. , 2., 2.]" --inpaint 


# ============== goal condition
## need to use fine-tuned model for each method
## must have condition
in feature_net.py, open 'cond_dim = 132'

models: 
grasp_mug_cond
grasp_bottle_cond
grasp_hammer_cond
grasp_fork_cond


python scripts/sample/real_sample_test.py --n_grasps 10  --model 'grasp_bottle_cond' \
--pc_path '../scene_data/scene_bottle_pc.npy' --save_dir '../temp_save/' --show --method 'goal'\
 --cond "[0. , 1.0, 0.]" 





