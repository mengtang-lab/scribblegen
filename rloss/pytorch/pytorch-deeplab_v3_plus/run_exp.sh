nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 0  &> nohup_logs/rloss_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data --gpu-ids 0  &> nohup_logs/rloss_aug_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 1  &> nohup_logs/partial_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data --gpu-ids 1  &> nohup_logs/partial_aug_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 3  &> nohup_logs/full_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data --gpu-ids 3  &> nohup_logs/full_aug_60.log &
sleep 3


nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 1 --gpu-ids 0  &> nohup_logs/rloss_aug_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 1 --aug-use-all --gpu-ids 0  &> nohup_logs/rloss_aug_1_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 2 --gpu-ids 1  &> nohup_logs/rloss_aug_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 2 --aug-use-all --gpu-ids 1  &> nohup_logs/rloss_aug_2_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 4 --gpu-ids 2  &> nohup_logs/rloss_aug_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 4 --aug-use-all --gpu-ids 2  &> nohup_logs/rloss_aug_4_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 8 --gpu-ids 3  &> nohup_logs/rloss_aug_8_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data normal --aug-ratio 8 --aug-use-all --gpu-ids 3  &> nohup_logs/rloss_aug_8_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data synth-only --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_synth_pre_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data synth-only --aug-ratio 2 --gpu-ids 3  &> nohup_logs/rloss_synth_pre_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data synth-only --aug-ratio 4 --gpu-ids 2  &> nohup_logs/rloss_synth_pre_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data synth-only --aug-ratio 8 --gpu-ids 1  &> nohup_logs/rloss_synth_pre_8_60.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 0 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_1_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 1 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_2_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 1 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_4_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 0 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_8_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_8_60.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data best --aug-ratio 1 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data best --aug-ratio 2 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data best --aug-ratio 4 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-data best --aug-ratio 8 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_8_60.log &