
"""
This module provides functionality to convert HDF5 image data into TIFF format and generate an ImageJ macro
for multi-emitter fitting in ThunderSTORM/imageJ.
"""

import numpy as np
import h5py
import tifffile
import os
import pathlib

if __name__ == '__main__':

    print("Loading generated data...")
    with h5py.File('input_for_cnn_star.h5', 'r') as h5f:
        images = np.array(h5f['star_vs_snr'])

    print("Converting generated images to tiff...")
    if not os.path.isdir('synth_data_tiff'):
        os.mkdir('synth_data_tiff')

    if not os.path.isdir('thunderstorm_outputs'):
        os.mkdir('thunderstorm_outputs')

    mxm = np.max(images)
    for framenum, frame in enumerate(images):
        frame_16bit = (frame*65_000/mxm).astype(np.uint16)
        path = f'synth_data_tiff/img_{framenum:d}.tiff'
        print(path)
        with tifffile.TiffWriter(path, imagej=True) as writer:
            writer.write(frame_16bit)
    
    print("Writing imageJ macro...")
    images_path = pathlib.Path(os.getcwd()).as_posix()
    
    #Optional generation of imageJ macro
    macro_template = """
    open("{path}/synth_data_tiff/{fn_in}");
    selectImage("{fn_in}");
    run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=false nmax=5 fixed_intensity=false pvalue=1.0E-6 renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
    run("Export results", "filepath={path}/thunderstorm_outputs/{fn_out} fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
    close;
    if (isOpen("Results")) {{
            selectWindow("Results");
            run("Close" );
    }};
    if (isOpen("Log")) {{
            selectWindow("Log");
            run("Close" );
    }};
    while (nImages()>0) {{
            selectImage(nImages());
            run("Close");
    }};
    """

    with open('MEFimageJmacro.txt', 'w') as mf:
        n = images.shape[0]
        for i in range(n):
            print(i)
            fn_in = f'img_{i:d}.tiff'
            fn_out = f'img_{i:03d}.csv'
            txt = (macro_template.format(fn_in = fn_in, fn_out = fn_out, path = images_path))
            mf.write(txt)
    print("Done")
