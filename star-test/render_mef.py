"""This script processes localization data from storm tables to generate 2D images and saves them 
as a dataset in an HDF5 file. The script is designed to handle multiple storm tables, rendering 
each into a corresponding 2D image frame.

The main functionality includes:
- Reading storm tables from CSV files.
- Converting localization data into pixel-based 2D images using a fixed scaling factor.
- Handling edge cases such as empty storm tables or single-row tables.
- Saving the generated image frames into an HDF5 file for further use.

Key Features:
- Localization data is mapped to pixel coordinates using a scaling factor of 1/60 nm/px.
- Intensities and offsets of overlapping localizations are summed for each pixel.
- Warnings are printed if multiple localizations overwrite the same pixel.
- The output is a 3D array of shape (n, h, w), where `n` is the number of storm tables, 
    and `h` and `w` are the height and width of the images.

Usage:
- Ensure the input HDF5 file (`input_for_cnn_star.h5`) contains the dataset `star_vs_snr` 
    with the number of storm tables.
- Place the storm table CSV files in the `thunderstorm_outputs/` directory, named as 
    `img_000.csv`, `img_001.csv`, etc.
- Run the script to generate the rendered images and save them in `mef_render.h5`.

Dependencies:
- numpy
- h5py

Output:
- An HDF5 file (`mef_render.h5`) containing the dataset `star_vs_snr` with the rendered images.
"""

import numpy as np
import h5py

def generate_storm_image(storm_table, h, w):    
    """
    Generates a 2D image from a storm table by mapping localizations to pixel coordinates.

    Parameters:
        storm_table (iterable): An iterable containing localization data. Each entry is expected 
                                to be a tuple with the following fields:
                                ("id", "frame", "x [nm]", "y [nm]", "sigma [nm]", 
                                "intensity [photon]", "offset [photon]", "bkgstd [photon]", 
                                "chi2", "uncertainty [nm]").
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
    kappa = 1/60 #nm/px of object space
    #read storm table
    for id, frame, x, y, sigma, intensity, offset, bkgstd, _a, _b in storm_table:
        #fixed upscale factor 4 here, factor 1.5 is due to grid fitting
        xnear = int(np.round(x*4*kappa)+1.5)
        ynear = int(np.round(y*4*kappa)+1.5)
        if (xnear >= w) or (xnear < 0) or (ynear >= h) or (ynear < 0):
            continue
        if img[ynear, xnear] != 0:
            print("Ooops!", xnear, ynear, 'is already there')
        img[ynear, xnear] += (intensity + offset)
    return img

if __name__ == '__main__':
    with h5py.File('input_for_cnn_star.h5', 'r') as h5f:
        n = h5f['star_vs_snr'].shape[0]

    print(f'Rendering {n} MEF tables into frames.')
    stormdata = np.zeros((n,200,200), dtype=np.float32)    
    for i in range(n):
        path = f'thunderstorm_outputs/img_{i:03d}.csv'        
        storm_table = np.genfromtxt(path, skip_header=1, delimiter=',')
        shape = storm_table.shape
        if storm_table.size == 0:
            print(f"img {i}: no emitters found, skipping render, keeping zeros")            
            continue
        elif len(shape)==1:
            storm_table = storm_table.reshape((1,-1))
        stormdata[i,:,:] = generate_storm_image(storm_table, 200, 200)

    print(f'Saving to mef_render.h5...')
    with h5py.File('mef_render.h5', 'w') as h5f:
        h5f.create_dataset('star_vs_snr', data = stormdata.astype(np.float32), dtype=np.float32)
    print("Done")