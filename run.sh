python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_full --add_phys_cam --cond full --defocus_type gaussian --learning_rate 0.0250 0.0030
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_wo_defocus --add_phys_cam --cond wo_defocus --learning_rate 0.0250 0.0030
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_wo_expsr --add_phys_cam --cond wo_expsr --defocus_type gaussian --learning_rate 0.0250 0.0030
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_wo_camera --learning_rate 0.0250 0.0030
