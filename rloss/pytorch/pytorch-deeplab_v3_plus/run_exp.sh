nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 0  &> nohup_logs/rloss_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme --gpu-ids 0  &> nohup_logs/rloss_aug_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 1  &> nohup_logs/partial_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme --gpu-ids 1  &> nohup_logs/partial_aug_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 3  &> nohup_logs/full_60.log &
sleep 3
nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 0 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme --gpu-ids 3  &> nohup_logs/full_aug_60.log &
sleep 3


nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 1 --gpu-ids 0  &> nohup_logs/rloss_aug_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 1 --aug-use-all --gpu-ids 0  &> nohup_logs/rloss_aug_1_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 2 --gpu-ids 1  &> nohup_logs/rloss_aug_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 2 --aug-use-all --gpu-ids 1  &> nohup_logs/rloss_aug_2_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 4 --gpu-ids 2  &> nohup_logs/rloss_aug_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 4 --aug-use-all --gpu-ids 2  &> nohup_logs/rloss_aug_4_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 8 --gpu-ids 3  &> nohup_logs/rloss_aug_8_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-ratio 8 --aug-use-all --gpu-ids 3  &> nohup_logs/rloss_aug_8_all_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme synth-only --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_synth_pre_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme synth-only --aug-ratio 2 --gpu-ids 3  &> nohup_logs/rloss_synth_pre_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme synth-only --aug-ratio 4 --gpu-ids 2  &> nohup_logs/rloss_synth_pre_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme synth-only --aug-ratio 8 --gpu-ids 1  &> nohup_logs/rloss_synth_pre_8_60.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 0 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_1_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 1 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_2_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 1 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_4_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --lr-start 60 --workers 6 --epochs 120 --scribbles --batch-size 12 --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --gpu-ids 0 --resume /home/jacob/scribblegen/rloss/pytorch/pytorch-deeplab_v3_plus/run/pascal/deeplab-mobilenet/rloss_synth_pre_8_60/checkpoint_epoch_60.pth.tar  &> nohup_logs/rloss_synth_post_8_60.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme best --aug-ratio 1 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme best --aug-ratio 2 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_2_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme best --aug-ratio 4 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_4_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme best --aug-ratio 8 --aug-best-dict best_synthetic_0_6.json --gpu-ids 3  &> nohup_logs/rloss_aug_best_8_60.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset hq --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_hq_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset cfg_no_text --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_aug_cfg_no_text_1_60.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset cfg_w_text --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_aug_cfg_w_text_1_60.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_1_25_n_50 --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_s_1_25_n_50.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_1_5_n_50 --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_s_1_5_n_50.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_1_75_n_50 --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_s_1_75_n_50.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_s_2_n_100.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_50 --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_aug_s_2_n_50.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_3_n_50 --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_aug_s_3_n_50.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_4_n_50 --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_aug_s_4_n_50.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_3_n_100 --aug-ratio 1 --gpu-ids 3  &> nohup_logs/rloss_aug_s_3_n_100.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset sig_04 --aug-ratio 1 --gpu-ids 0  &> nohup_logs/rloss_aug_sig_04.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset sig_05 --aug-ratio 1 --gpu-ids 0  &> nohup_logs/rloss_aug_sig_05.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset sig_06 --aug-ratio 1 --gpu-ids 1  &> nohup_logs/rloss_aug_sig_06.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset sig_07 --aug-ratio 1 --gpu-ids 1  &> nohup_logs/rloss_aug_sig_07.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset sig_08 --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_sig_08.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset sig_09 --aug-ratio 1 --gpu-ids 2  &> nohup_logs/rloss_aug_sig_09.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.1 --gpu-ids 1  &> nohup_logs/rloss_replace_01.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.2 --gpu-ids 1  &> nohup_logs/rloss_replace_02.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.3 --gpu-ids 1  &> nohup_logs/rloss_replace_03.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.4 --gpu-ids 1  &> nohup_logs/rloss_replace_04.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.5 --gpu-ids 1  &> nohup_logs/rloss_replace_05.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.6 --gpu-ids 2  &> nohup_logs/rloss_replace_06.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.7 --gpu-ids 2  &> nohup_logs/rloss_replace_07.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.8 --gpu-ids 2  &> nohup_logs/rloss_replace_08.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme replacement --aug-dataset s_2_n_100 --aug-ratio 1 --replacement-prob 0.9 --gpu-ids 2  &> nohup_logs/rloss_replace_09.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset normal --aug-ratio 1 --gpu-ids 2 --curriculum '{"0": "sig_05", "10": "sig_06", "20": "sig_07", "30": "sig_08", "40": "sig_09", "50": "normal"}'  &> nohup_logs/rloss_curriculum.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.003 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset normal --aug-ratio 1 --gpu-ids 1 --curriculum '{"0": "sig_05", "10": "sig_06", "20": "sig_07", "30": "sig_08", "40": "sig_09", "50": "normal"}'  &> nohup_logs/rloss_curriculum_3.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.001 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset normal --aug-ratio 1 --gpu-ids 2 --curriculum '{"0": "sig_05", "10": "sig_06", "20": "sig_07", "30": "sig_08", "40": "sig_09", "50": "normal"}'  &> nohup_logs/rloss_curriculum_1.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.003 --workers 6 --epochs 120 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset normal --aug-ratio 1 --gpu-ids 1 --curriculum '{"0": "sig_05", "20": "sig_06", "40": "sig_07", "60": "sig_08", "80": "sig_09", "100": "normal"}'  &> nohup_logs/rloss_curriculum_3.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.001 --workers 6 --epochs 120 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset normal --aug-ratio 1 --gpu-ids 2 --curriculum '{"0": "sig_05", "20": "sig_06", "40": "sig_07", "60": "sig_08", "80": "sig_09", "100": "normal"}'  &> nohup_logs/rloss_curriculum_1.log &



nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 1_2 --gpu-ids 1  &> nohup_logs/rloss_split_1_2.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 1_4 --gpu-ids 1  &> nohup_logs/rloss_split_1_4.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 1_8 --gpu-ids 1  &> nohup_logs/rloss_split_1_8.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 1_16 --gpu-ids 1  &> nohup_logs/rloss_split_1_16.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 1464 --gpu-ids 2  &> nohup_logs/rloss_split_1464.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 732 --gpu-ids 2  &> nohup_logs/rloss_split_732.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 366 --gpu-ids 2  &> nohup_logs/rloss_split_366.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 183 --gpu-ids 2  &> nohup_logs/rloss_split_183.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 0_7_p --gpu-ids 1  &> nohup_logs/rloss_split_0_7_p.log &

nohup python train_withdensecrfloss.py --backbone mobilenet --lr 0.007 --workers 6 --epochs 60 --scribbles --batch-size 12  --checkname deeplab-mobilenet --eval-interval 2 --dataset pascal --save-interval 5 --densecrfloss 2e-9 --rloss-scale 0.5 --sigma-rgb 15 --sigma-xy 100 --aug-scheme normal --aug-dataset s_2_n_100 --aug-ratio 1 --ssl-split 0_9_p --gpu-ids 2  &> nohup_logs/rloss_split_0_9_p.log &

