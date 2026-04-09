 Train Q3 model
 %cd /kaggle/working/Computer-Vision_Assignments_Lakshay/Q3
!python train.py --config config/default.yaml --data_root "$DATA_ROOT"

7) Run Q3 analysis (permutation + critical points + sparse robustness)
%cd /kaggle/working/Computer-Vision_Assignments_Lakshay/Q3
!python analysis.py --config config/default.yaml --data_root "$DATA_ROOT" --checkpoint outputs/q3_pointnet/best.pt

8) Check required outputs
!find /kaggle/working/Computer-Vision_Assignments_Lakshay/Q3/outputs -maxdepth 3 -type f | sort