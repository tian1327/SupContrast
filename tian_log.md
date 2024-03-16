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

# use CLIP-SSL to train the linear classifier for stage 3
cd ../CLIP-SSL

python main.py --model_path ../SupContrast/save/SupCon/semi-aves_models/no-color-aug_SupCon_semi-aves_vitb32_openclip_laion400m_lr_1e-06_decay_0.01_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_10.pth --epochs 10 --prefix stage3-LP-FT-SCL-fs15 --method probing --bsz 32 --train_split fewshot15.txt --folder output_stage3 --pre_extracted False

```
2. remove color augmentation, add a projection layer.
```bash
python main_supcon_clip.py --prefix no-color-aug --batch_size 256   --epochs 1   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --temp 0.1   --cosine   --save_freq 1

# use CLIP-SSL to train the linear classifier for stage 3
cd ../CLIP-SSL

python main.py --model_path ../SupContrast/save/SupCon/semi-aves_models/no-color-aug-mlp-project_SupCon_semi-aves_vitb32_openclip_laion400m_lr_1e-06_decay_0.01_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_1.pth --epochs 10 --prefix stage3-LP-FT-SCL-fs15 --method probing --bsz 32 --train_split fewshot15.txt --folder output_stage3 --pre_extracted False
```

3. run few-shot anchored SCL
```bash
python main_supcon_clip.py --prefix FASupCon --batch_size 256   --epochs 100   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --temp 0.1   --cosine   --save_freq 10 --method FASupCon

# use CLIP-SSL to train the linear classifier for stage 3
cd ../CLIP-SSL

python main.py --model_path ../SupContrast/save/SupCon/semi-aves_models/FASupCon_FASupCon_semi-aves_vitb32_openclip_laion400m_lr_1e-06_decay_0.01_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_40.pth --epochs 10 --prefix stage3-FA-SCL --method probing --bsz 32 --train_split fewshot15.txt --folder output_stage3 --pre_extracted False

# run for 1 epoch
python main_supcon_clip.py --prefix FASupCon --batch_size 256   --epochs 1   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --temp 0.1   --cosine   --save_freq 1 --method FASupCon

```

20240315 Friday
1. Run SCL on balanced few-shot data, see how well it performs, if comparable with the FT-CE finetuned with few-shot data
```bash
# check tian_log.md in SupContrast

# vanilla SCL
python main_supcon_clip.py --prefix SupCon-fs15 --batch_size 256   --epochs 20   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15.txt   --temp 0.1   --cosine   --save_freq 5 --method SupCon

# FA-SCL
python main_supcon_clip.py --prefix FASupCon-fs15 --batch_size 256   --epochs 20   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15.txt   --temp 0.1   --cosine   --save_freq 5 --method FASupCon

# test stage 3
cd ../CLIP-SSL

# FA-SCL
python main.py --model_path ../SupContrast/save/SupCon/semi-aves_models/FASupCon_FASupCon_semi-aves_vitb32_openclip_laion400m_lr_1e-06_decay_0.01_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_1.pth  --epochs 10 --prefix stage3-LP-FT-SCL-fs15 --method probing --bsz 32 --lr_classifier 1e-5 --train_split fewshot15.txt --folder output_stage3 --pre_extracted False

# SCL
python main.py --model_path ../SupContrast/save/SupCon/semi-aves_models/SupCon-fs15_SupCon_semi-aves_vitb32_openclip_laion400m_lr_1e-06_decay_0.01_bsz_256_temp_0.1_trial_0_cosine/ckpt_epoch_1.pth --epochs 10 --prefix stage3-LP-FT-SCL-fs15 --method probing --bsz 32 --lr_classifier 1e-5 --train_split fewshot15.txt --folder output_stage3 --pre_extracted False

```