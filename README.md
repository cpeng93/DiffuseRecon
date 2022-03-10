## DiffuseRecon

This codebase is modified based on Improved DDPM - https://github.com/openai/improved-diffusion

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.


## Training

mpiexec -n GPU_NUMS python scripts/image_train.py --data_dir TRAIN_PATH --image_size 320 --num_channels 128\
 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 4000 --noise_schedule cosine --lr 1e-4 --batch_size 1\
--save_dir img_space_dual

## Sampling

python scripts/image_sample_complex_duo.py --model_path img_space_dual/ema_0.9999_150000.pt --data_path EVAL_PATH \
--image_size 320 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 4000 \
--noise_schedule cosine --timestep_respacing 100 --save_path test/ --num_samples 1 --batch_size 5

## More Details To Come