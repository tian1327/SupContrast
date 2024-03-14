2024/03/12 Tue
1. run Supervised Contrastive Learning on Semi-aves with OpenCLIP
```bash
python main_supcon_clip.py --batch_size 256   --epochs 100   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --temp 0.1   --cosine   --save_freq 1

# or use the slurm file
sbatch supcon.slurm

# use CLIP-SSL to train the linear classifier for stage 3
cd ../CLIP-SSL

python main.py --model_path ../SupContrast/save/SupCon/semi-aves_models/SupCon_semi-aves_vitb32_openclip_laion400m_lr_1e-06_decay_0.01_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_100.pth --epochs 10 --prefix stage3-LP-FT-SCL100-fs15 --method probing --bsz 32 --train_split fewshot15.txt --folder output_stage3 --pre_extracted False

```

20240314 Thursday
1. run SCL by removing the color augmentation of birds, stil no projection layer added
```bash
python main_supcon_clip.py --prefix no-color-aug --batch_size 256   --epochs 1   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --temp 0.1   --cosine   --save_freq 1
```
2. remove color augmentation, add a projection layer.
```bash
python main_supcon_clip.py --prefix no-color-aug --batch_size 256   --epochs 1   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --temp 0.1   --cosine   --save_freq 1
```
