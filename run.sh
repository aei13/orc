# Train Model with 2 GPU, 40 epoch, RI-M, L1+SPL, lambda=0.01, VGG16 conv4-1 layer setting
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_endpoint=localhost:5254 --nnodes=1 --nproc_per_node=2 main_2d.py --model restormer2d_fq_e_igateqkv --lr 3e-4 --weight-decay 1e-4 --batch-size 64 --epochs 40 --crop --input-size 64 --input_type RI --output_type M --loss_lda_l1 0.99 0.0 0.0 0.0 --loss_lda_per 0.01 0.0 0.0 0.0 --idx_layers_per 9 --loss_per_type mul --clip-grad 1.0 --output_dir "./experiments/restormer2d_fq_e_igateqkv" --data-set "MRI2D" --data-path "./data/mri"

# Train Model with 2 GPU, 40 epoch, RI-RI, L1 (lambda=0)
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_endpoint=localhost:5254 --nnodes=1 --nproc_per_node=2 main_2d.py --model restormer2d_fq_e_igateqkv --lr 3e-4 --weight-decay 1e-4 --batch-size 64 --epochs 40 --crop --input-size 64 --input_type RI --output_type RI --loss_lda_l1 0.5 0.5 0.0 0.0 --loss_lda_per 0.0 0.0 0.0 0.0 --clip-grad 1.0 --output_dir "./experiments/restormer2d_fq_e_igateqkv" --data-set "MRI2D" --data-path "./data/mri"


# Inference
CUDA_VISIBLE_DEVICES=0 python3 main_2d.py --eval --model restormer2d_fq_e_igateqkv --batch-size 1 --input_type RI --output_type M --output_dir "./experiments/restormer2d_fq_e_igateqkv" --data-set "MRI2D" --data-path "./data/mri" --pretrained './experiments/restormer2d_fq_e_igateqkv/checkpoint.pth'

# Inference with TTM (4 pixel, median)
CUDA_VISIBLE_DEVICES=0 python3 main_2d.py --eval --model restormer2d_fq_e_igateqkv --batch-size 1 --input_type RI --output_type M --translation_type 'all' --translation_iter 4 --translation_fill_type median --output_dir "./experiments/restormer2d_fq_e_igateqkv" --data-set "MRI2D" --data-path "./data/mri" --pretrained './experiments/restormer2d_fq_e_igateqkv/checkpoint.pth'