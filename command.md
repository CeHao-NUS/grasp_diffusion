
# Object Classes :['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
# 'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
# 'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
# 'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
# 'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
# 'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']



Mug
Teacup
WineBottle
Bottle



ScrewDriver
Fork
Hammer
Knife
Scissors
Spoon



# ============= hammer ==============
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class 'Hammer' 


--eval_sim True

# =========== mug ==============
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 1 --obj_class 'Mug' --model 'grasp_dif_mugs'





main: 
sample: Grasp_AnnealedLD
diff model: GraspDiffusionFields