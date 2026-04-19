Cell 3 — Vanilla
%cd /kaggle/working/Computer-Vision_Assignments_Lakshay/Q2
!python train.py --config config/default.yaml --variant vanilla --data_root "$DATA_ROOT" --epochs 10

Cell 4 — No-skip
%cd /kaggle/working/Computer-Vision_Assignments_Lakshay/Q2
!python train.py --config config/default.yaml --variant noskip --data_root "$DATA_ROOT" --epochs 10

Cell 5 — Residual
%cd /kaggle/working/Computer-Vision_Assignments_Lakshay/Q2
!python train.py --config config/default.yaml --variant residual --data_root "$DATA_ROOT" --epochs 10

3) Evaluate on test set (mIoU + RMSE + qualitative)

%cd /kaggle/working/Computer-Vision_Assignments_Lakshay/Q2
!python eval.py --config config/default.yaml --variant vanilla  --data_root "$DATA_ROOT" --checkpoint "outputs/q2_multitask_unet_vanilla/best.pt"
!python eval.py --config config/default.yaml --variant noskip   --data_root "$DATA_ROOT" --checkpoint "outputs/q2_multitask_unet_noskip/best.pt"
!python eval.py --config config/default.yaml --variant residual --data_root "$DATA_ROOT" --checkpoint "outputs/q2_multitask_unet_residual/best.pt"

