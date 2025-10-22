import sys

import decode
import decode.utils

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

import h5py

import scipy.ndimage as ndi

def l1norm(a):
    asu = a - np.min(a)
    norm = np.sum(asu)
    if norm == 0:
        return np.zeros_like(a)
    return asu/norm

def upscale_and_round_to_grid(x, y):
    kappa = 1
    xnear = int(np.round(x*4*kappa)+1.5) + 5 #these values are here to compensate systematic shift of the DECODE
    ynear = int(np.round(y*4*kappa)+1.5) + 6
    #return xnear, ynear
    return ynear, xnear

def generate_decode_image(decode_xyi, h, w, shift_foo):    
    """
    Generates a 2D image from a storm table by mapping localizations to pixel coordinates.

    Parameters:
        decode_xyi ... TODO
        h (int): Height of the output image in pixels.
        w (int): Width of the output image in pixels.

    Returns:
        numpy.ndarray: A 2D array of shape (h, w) representing the generated image, where each 
                       pixel value corresponds to the summed intensity and offset of localizations 
                       mapped to that pixel.

    Notes:
        - The function assumes a fixed scaling factor (kappa = 1/60 nm/px) to convert 
          nanometer coordinates to pixel coordinates.
        - It works with fixed upscaling factor 4x
        - Localizations are rounded to the nearest pixel using nearest-neighbor rounding.
        - If a localization falls outside the image bounds, it is ignored.
        - If multiple localizations map to the same pixel, their intensities and offsets are summed.
        - A warning message is printed if a pixel is overwritten by multiple localizations.
    """
    #"id","frame","x [nm]","y [nm]","sigma [nm]","intensity [photon]","offset [photon]","bkgstd [photon]","chi2","uncertainty [nm]"
    img = np.zeros((h,w), dtype=np.float64)
    # kappa = 1#1/60 #nm/px of object space
    #read storm table
    for x, y, intensity in decode_xyi:
        #fixed upscale factor 4 here, factor 1.5 is due to grid fitting
        xnear, ynear = shift_foo(x, y)
        if (xnear >= w) or (xnear < 0) or (ynear >= h) or (ynear < 0):
            continue
        if img[ynear, xnear] != 0:
            print("Ooops!", xnear, ynear, 'is already there')
        img[ynear, xnear] += (intensity)
    return img

if __name__ == '__main__':
    print(f"DECODE version: {decode.utils.bookkeeping.decode_state()}")
    #Setup
    device = 'cuda:0'  # or 'cpu', or you change cuda device index
    threads = 4  #  number of threads, useful for CPU heavy computation. Change if you know what you are doing.
    worker = 0  # number of workers for data loading. Used a default of 0 for safety. Multiprocessing on windows is sometimes not stable
    
    torch.set_num_threads(threads)  # set num threads
    
    if device != 'cpu':
        if not torch.cuda.is_available():
            raise ValueError("You have selected a non CPU device, but CUDA is not available."
                             "Refer to CPU version or check your installation.")
    
    # here you need to specify the parameters with suffix _run.yaml in your model's output folder (not param_run_in.yaml)
    param_path = '../../decode-model/gauss_snr_74/param_run.yaml'
    model_path = '../../decode-model/gauss_snr_74/model_0.pt'
    meta = {
        'Camera': {
            'baseline': 50,
            'e_per_adu': 1.0,
            'em_gain': 1.0,
            'spur_noise': 0.000,  # if you don't know, you can set this to 0.
            'convert_photons' : False,
            'px_size': (60,60)
        }
    }
    param = decode.utils.param_io.load_params(param_path)
    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(model,
                                                input_file=model_path,
                                                output_file=None).load_init(device=device)
    # overwrite camera
    param = decode.utils.param_io.autofill_dict(meta['Camera'], param.to_dict(), mode_missing='include')
    param = decode.utils.param_io.RecursiveNamespace(**param)    
    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = 'cpu'
    
    #setup 2
    frame_proc = decode.neuralfitter.utils.processing.TransformSequence([
        # decode.neuralfitter.utils.processing.wrap_callable(camera.backward), #no conversion from photon to ADU
        decode.neuralfitter.frame_processing.AutoCenterCrop(8),
        decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param) #keep rescaling
    ])
    
    
    # determine extent of frame and its dimension after frame_processing
    frames = torch.rand((10,50,50))
    size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(), frame_proc.forward)  # frame size after processing
    frame_extent = ((-0.5, size_procced[-2] - 0.5), (-0.5, size_procced[-1] - 0.5))
    
    # Setup post-processing
    # It's a sequence of backscaling, relative to abs. coord conversion and frame2emitter conversion
    post_proc = decode.neuralfitter.utils.processing.TransformSequence([
        decode.neuralfitter.scale_transform.InverseParamListRescale.parse(param),
        decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0],
                                                              yextent=frame_extent[1],
                                                              img_shape=size_procced[-2:]),
    
        decode.neuralfitter.post_processing.SpatialIntegration(raw_th=0.1,
                                                              xy_unit='px',
                                                              px_size=param.Camera.px_size)
    ])
    
    #instantiate predictor object
    infer = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
                                      frame_proc=frame_proc, post_proc=post_proc,
                                      device=device, num_workers=worker)    
    def em_to_xyi(em_obj):
        xyz = em_obj.xyz
        phot = em_obj.phot
        print(len(xyz), 'em')
        xyi = np.zeros((xyz.shape[0], 3))
        xyi[:,0:2] = xyz[:,0:2]
        xyi[:,2] = phot
        return xyi

    #----------------------
    INPUTS = [f'mc_img{i}.h5' for i in (1, 2, 3)]
    OUTPUTS = [f'mc_img{i}_decode2.h5' for i in (1, 2, 3)]
    
    for file_in, file_out in zip(INPUTS, OUTPUTS):
        with h5py.File(file_in, 'r') as h5fi, h5py.File(file_out, 'w') as h5fo:
            for key in list(h5fi.keys()):
                print(key, "...")
                rendered_images = []                
                gen = ((i, img) for i, img in enumerate(h5fi[key]))
                for i, img in gen:
                    # frames = torch.from_numpy(img.astype(np.float32).reshape((1,50,50)))
                    # _img = l1norm(img)
                    _img = img - np.min(img)
                    _img = _img/np.sum(img)
                    _img = np.clip(_img * 7e5, 0, None)
                    # _img = np.clip(_img * 1e5, 0, None)
                    frames = torch.from_numpy(_img.astype(np.float32).reshape((1,50,50)))
                    emitters = infer.forward(frames)
                    print(emitters.xyz)
                    rendered = (generate_decode_image(em_to_xyi(emitters), 200, 200, upscale_and_round_to_grid))
                    if len(emitters.xyz)>0:
                        rendered = l1norm(rendered)
                    rendered_images.append(rendered)

                h5fo.create_dataset(key, data = np.array(rendered_images))
    print("Done.")
    
    