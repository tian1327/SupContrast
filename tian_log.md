2024/03/06 Wed
1. run Supervised Contrastive Learning on Semi-aves with OpenCLIP
```bash
python main_supcon.py --batch_size 256 \
  --epochs 1 \
  --learning_rate 1e-6 \
  --weight_decay 1e-2 \
  --dataset semi-aves \
  --data_folder ../CLIP-SSL/data/semi-aves/ \
  --train_split fewshot15+real_t2t500.txt \
  --size 224 \
  --temp 0.1 \
  --cosine \
  --warm

# or use the slurm file
sbatch supcon.slurm

```