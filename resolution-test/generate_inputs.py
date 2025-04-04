# Generate synthetic data for resolution testing.
# Outputs will be be processed by process_inputs.py

import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import h5py

#Parameters of simulation
RNG = np.random.default_rng(seed=20250113)
MONTE_CARLO_SAMPLES = 10
BCKG = 3
#Width of the PSF exp[-(x-mu)^2 / (2 sigma^2)]
PSFW = 2
#approximate rayleigh limit distance, obtained by approximating Airy pattern with gaussian.
RAYLEIGH = 3*PSFW 
#Image size
WIDTH = 50
HEIGTH = 50
BCKG = 3 #background level (offset)
UPSCALE = 4 #upscaling factor for rendering of reference images
#range of tested SNRs
snrs = np.logspace(0, 3, 37) 
#generate emitters positions by
#increasing its distance in each frame
distances = np.linspace(0,1,41)*RAYLEIGH
xys = np.array([((25-d/2, 25),(25+d/2, 25)) for d in distances])

snrs = list(snrs) + [-1] #-1 is a special SNR and mean no noise

# ------------------------------------------
def gauss_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def generate_image(xyi, noise_mean, noise_std, psf_width, h, w):    
    img = RNG.normal(noise_mean, noise_std, h*w).reshape(h,w)
    ys = np.arange(h)
    xs = np.arange(w)
    xy = np.meshgrid(xs, ys)
    for x, y, intensity in xyi:
        img += gauss_2d(xy, intensity, x, y, psf_width, psf_width, 0, 0).reshape((h,w))
    img = (img - np.min(img))
    img = img/np.sum(img)
    return img

def generate_gt_image(xyi, h, w):    
    img = np.zeros((h, w))
    for x, y, intensity in xyi:
        x_nearest = int(np.round(x))
        y_nearest = int(np.round(y))        
        if (x_nearest <= w-1) and (y_nearest <= h-1):
            img[y_nearest, x_nearest] += intensity
    return img/np.sum(img)   
  
# ------------------------------------------
snrs_mc = np.kron(snrs,np.ones(MONTE_CARLO_SAMPLES))
datasets_inp = dict()
datasets_ref = dict()

MONTE_CARLO_SAMPLES = 10
PSFW = 2
RAYLEIGH = 3*PSFW

WIDTH = 50
HEIGTH = 50
UPSCALE = 4
snrs = np.logspace(0, 3, 37)

distances = np.linspace(0,1,41)*RAYLEIGH
xys = np.array([((25-d/2, 25),(25+d/2, 25)) for d in distances])

snrs = list(snrs) + [-1]
snrs_mc = np.kron(snrs,np.ones(MONTE_CARLO_SAMPLES))

datasets_inp = dict()
datasets_ref = dict()
            # imgref = generate_gt_image(upscale_xyi(xyi), 200, 200)
            # images_ref.append(imgref)

for snr in snrs_mc:
    key = f'snr_{int(round(snr)):d}_{counter}' if snr>0 else 'noiseless'
    print(key, snr)
    #background level offset is 1 for all standard SNRs
    #but when we prepare noiseless data, background level is 0, SNR is set to 1 and no noise is applied.
    _bckgstd = 1 if snr>0 else 0
    _snr = snr if snr>0 else 1 
    images_in = []
    for xy in xys:  
        xyi = np.vstack([xy[:,0], xy[:,1], np.ones(2)*_snr]).T
        images_in.append(generate_image(xyi, 3, _bckgstd, PSFW, 50, 50))
    datasets_inp[key] = np.array(images_in)    
    
key = 'reference'
images_ref = []
lateral_offset = (UPSCALE-1)/2
for xy in xys:
    xyi = np.vstack([UPSCALE*xy[:,0]+lateral_offset, UPSCALE*xy[:,1]+lateral_offset, np.ones(2)*1]).T    
    images_ref.append(generate_gt_image(xyi, 200, 200))

datasets_ref[key] = np.array(images_ref)

print("Writing inputs...")
with h5py.File('input_res_for_cnn_psf4.h5', 'w') as h5f:
    for key in datasets_inp.keys():
        print(key)
        h5f.create_dataset(key, data = datasets_inp[key].astype(np.float32), dtype=np.float32)

print("Writing references...")
with h5py.File('synth_res_reference_psf4.h5', 'w') as h5f:
    for key in datasets_ref.keys():
        print(key)
        h5f.create_dataset(key, data = datasets_ref[key].astype(np.float32), dtype=np.float32)
print("Done")
