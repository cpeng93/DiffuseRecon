import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import pickle
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util_duo import (
    model_and_diffusion_defaults,
    create_model_and_two_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import imageio
mask = pickle.load(open('file1000031_mask.pt','rb')).view(1,1,320,320).cuda()
mask= th.cat([mask,mask],1)
images = ['file1000031']

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_path)

    logger.log("creating model and diffusion...")
    model, diffusion, diffusion_two = create_model_and_two_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    for image in images:
        slice = 16
        coarse = []
        for i in range(slice-1,slice+2):
            file_name1 = image + '_' + str(i) + '.pt'
            file_name2 = image + '_' + str(i + 1) + '.pt'
            kspace = load_data(args.data_path,file_name1,file_name2,args.batch_size)
            #save for refining
            if i == slice:
                input = kspace[[0]]
            logger.log("sampling...")
            samples = []
            for _ in range(2):
                model_kwargs = {}
                sample = diffusion.p_sample_loop_condition(
                    model,
                    (args.batch_size, 4, args.image_size, args.image_size),
                    kspace,
                    mask,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs
                )[-1]
                samples.append(sample)
            samples = th.cat(samples)
            coarse.append(samples.contiguous())

        coarse = th.stack(coarse)
        print(coarse.shape)
        aggregate = []
        for k in range(2):
            aggregate.append((coarse[k,:,[2,3]].mean(0) + coarse[k+1,:,[0,1]].mean(0)).view(1,2,320,320)/2)
        aggregate = th.cat(aggregate,1)
        print(aggregate.shape)

        sample2 = diffusion_two.p_sample_loop_condition(
            model,
            (1, 4, args.image_size, args.image_size),
            input,
            mask,
            noise=aggregate.float(),
            clip_denoised=args.clip_denoised,
            model_kwargs={},
            refine=True
        )
        sample2 = sample2[-1].cpu().data.numpy()
        pickle.dump({'coarse':coarse.cpu().data.numpy(),'fine':sample2},
                    open(os.path.join(args.save_path,image+'_'+str(slice)+'.pt'),'wb'))
        vis = np.abs(sample2[0,0]+sample2[0,1]*1j)
        imageio.imsave(os.path.join(args.save_path,image+'_'+str(slice)+'.png'),vis/vis.max())

def load_data(data_path, file1, file2, batch_size):
    # load two slices
    img_prior1 = pickle.load(open(os.path.join(data_path, file1), 'rb'))['img']
    img_prior2 = pickle.load(open(os.path.join(data_path, file2), 'rb'))['img']
    print('loading', file1, file2)
    data = np.stack([np.real(img_prior1), np.imag(img_prior1), np.real(img_prior2), np.imag(img_prior2)]).astype(
        np.float32)
    max_val = abs(data[:2]).max()
    data[:2] /= max_val
    max_val = abs(data[2:4]).max()
    data[2:4] /= max_val
    # regularizing over max value ensures this model works over different preprocessing schemes;
    # to not use the gt max value, selecting an appropriate averaged max value from training set leads to
    # similar performance, e.g.
    # data /= 7.21 (average max value); in general max_value is at DC and should be accessible.
    data1 = data[0] + data[1] * 1j
    data2 = data[2] + data[3] * 1j
    kspace1 = np.fft.fft2(data1)
    kspace2 = np.fft.fft2(data2)
    kspace = th.FloatTensor(
        np.stack([np.real(kspace1), np.imag(kspace1), np.real(kspace2), np.imag(kspace2)])) \
        .cuda().view(1, 4, 320, 320).repeat(batch_size, 1, 1, 1).float()
    return kspace

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=5,
        use_ddim=False,
        model_path="",
        data_path="",
        save_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
