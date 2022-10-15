## DiffuseRecon

This codebase is modified based on [Improved DDPM](https://github.com/openai/improved-diffusion)

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## Data Preparation and Pre-Trained Checkpoints

A pre-trained checkpoint can be downloaded via this [link](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/cpeng26_jh_edu/EaZmOZRiAYVPgv8H1AdpszkBcvN6mWqqPhm0KR0owWEFjw?e=oXAqTa).

For FastMRI, the simplified h5 data can be downloaded by following the instructions in [ReconFormer](https://github.com/guopengf/ReconFormer), i.e. through [Link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/pguo4_jh_edu/EtXsMeyrJB1Pn-JOjM_UqhUBdY1KPrvs-PwF2fW7gERKIA?e=uuBINy). DiffuseRecon converts it to a normalized format in scripts/data_process.py

```
python scripts/data_process.py
```



## Sampling

```
python scripts/image_sample_complex_duo.py --model_path img_space_dual/ema_0.9999_150000.pt --data_path EVAL_PATH \
--image_size 320 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 4000 \
--noise_schedule cosine --timestep_respacing 100 --save_path test/ --num_samples 1 --batch_size 5
```
Note that timestep_respacing indicates the initial coarse sampling steps. 
## Training

```
mpiexec -n GPU_NUMS python scripts/image_train.py --data_dir TRAIN_PATH --image_size 320 --num_channels 128\
 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --lr 1e-4 --batch_size 1\
--save_dir img_space_dual
```
## TODO
- Upload PSNR evaluation
- Currently, the refinement step is fixed at 20 (line 592, gaussian_diffusion_duo.py); make this an adjustable input.
- Graphics.