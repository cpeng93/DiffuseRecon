from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os, pickle

#provide your path for DiffuseRecon results, saved in pickle, in the dimension of of H,W, # of slice
duo_dir = 'kspace_duo_same_mask_all/vols/results/'
#provide your path for groundtruth
orig_gt_dir = '/cis/home/cpeng/mri_recon/T1/val/'
files = os.listdir(orig_gt_dir)
def norm(img):
    img -= img.mean()
    img /= img.std()
    return img

def normalize_complex(data, eps=0.):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std

psnr_DDPM_duo = []
for file in files:
	data = h5py.File(orig_gt_dir + file, 'r')['kspace']
	orig_target = []
	#the same as data_process, normed in the image space in the end
	for i in range(data.shape[0]):
	    norm_kspace,std_kspace = normalize_complex(data[i])
	    img = np.fft.ifft2(norm_kspace)
	    img = np.fft.fftshift(img)
	    norm_img,std_img = normalize_complex(img)
	    #norm in the image space
	    orig_target.append(norm(np.abs(norm_img)))

	orig_target = np.asarray(orig_target).transpose(1,2,0)
	DDPM_duo = pickle.load(open(duo_dir+file.replace('.h5','_full.pt'),'rb'))
	orig_target = orig_target[...,4:-1]
	DDPM_duo = DDPM_duo[...,4:-1]

	data_range = orig_target.max() - orig_target.min()
	psnr_DDPM_duo.append(psnr(orig_target[...,:min_overlap],DDPM_duo[...,:min_overlap],data_range=data_range))
	print(psnr_DDPM_duo[-1])

print(np.mean(psnr_DDPM_duo))