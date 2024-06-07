export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch --nproc_per_node=1 run.py --config configs/nusc-sem.txt --eval_only --load_weights_folder ckpts/nusc-sem