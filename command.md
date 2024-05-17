
# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']



Mug (1)
Teacup
WineBottle
Bottle (3)
Cup


ScrewDriver
Fork (5)
Hammer (1)
Knife
Scissors
Spoon



python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class Mug
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 3 --obj_class Bottle
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class Hammer
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 5 --obj_class Fork


# ============= hammer ==============
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 20 --obj_id 1 --obj_class 'Hammer' 


--eval_sim True

# =========== mug ==============
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 20 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_mugs'





main: 
sample: grasp_samplers/ Grasp_AnnealedLD
diff model: GraspDiffusionFields



# =================== train cond models ==========================
python scripts/train/train_pointcloud_6d_grasp_diffusion.py \
--class_type 'Mug' --exp_log_dir 'grasp_mug'

python scripts/train/train_pointcloud_6d_grasp_diffusion.py \
--class_type 'Bottle' --exp_log_dir 'grasp_bottle'

python scripts/train/train_pointcloud_6d_grasp_diffusion.py \
--class_type 'Hammer' --exp_log_dir 'grasp_hammer'

python scripts/train/train_pointcloud_6d_grasp_diffusion.py \
--class_type 'Fork' --exp_log_dir 'grasp_fork'

## train one

python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_mug_train'

## change
1. acronym_dataset.py / === manual select one ======
2. grasp_dif.py / cond_ext
3. sdf_loss.py / model.set_latent                                                           
4. denoising_loss.py / model.set_latent
5. feature_net.py / cond_dim = 132



python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_mug'


python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 20 --obj_id 3 --obj_class 'Bottle' --model 'grasp_bottle'

python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 20 --obj_id 1 --obj_class 'Hammer' --model 'grasp_hammer'

python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 5 --obj_class 'Fork' --model 'grasp_fork'

