import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile
import os

imagej_macro_template = """
open("{PATH}/thst_data/{fn_in}");
selectImage("{fn_in}");
run("Run analysis", "filter=[Wavelet filter (B-Spline)] scale=2.0 order=3 detector=[Local maximum] connectivity=8-neighbourhood threshold=std(Wave.F1) estimator=[PSF: Integrated Gaussian] sigma=1.6 fitradius=3 method=[Weighted Least squares] full_image_fitting=false mfaenabled=true keep_same_intensity=false nmax=5 fixed_intensity=false pvalue=1.0E-6 renderer=[Averaged shifted histograms] magnification=5.0 colorizez=false threed=false shifts=2 repaint=50");
run("Export results", "filepath={PATH}/thunderstorm_outputs/{fn_out} fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true id=true uncertainty=true frame=true");
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

if __name__ == '__main__':
    #use relative to CWD or replace this by an absolute path
    PATH = os.path.join(os.getcwd(), 'resampled').replace('\\', '/')

    thunderstorm_outputs_dir = os.path.join(PATH, 'thunderstorm_outputs').replace('\\', '/')
    thst_data_dir = os.path.join(PATH, 'thst_data').replace('\\', '/')
    print('Making dirs:', thunderstorm_outputs_dir, thst_data_dir)
    os.makedirs(thunderstorm_outputs_dir, exist_ok=True)
    os.makedirs(thst_data_dir, exist_ok=True)


    with open('imageJmacro.txt', 'w') as mf:
        mf.write('//Generated macro for ImageJ/ThunderStorm\n\n')

    ns = [1,2,3]

    for i in ns:
        print(f"Dataset: {i}")
        fn = f'resampled/mc_img{i}.h5'
        with h5py.File(fn, 'r') as h5f:
            images = np.array(h5f[f'resampled'])
            # img0 = np.array(h5f[f'original'])
        
        #rescale image from camera counts to to 16-bit TIFF
        images_16bit = ((images/np.max(images))*(2**16 - 1)).astype(np.uint16)

        #save all images
        for j, img in enumerate(images_16bit):
            fn_out = f"resampled/thst_data/dset{i}_frm{j}.tiff" 
            print(fn_out)
            with tifffile.TiffWriter(fn_out, imagej = True) as tffw:
                tffw.write(data = img)
            
                
        #fill entro for imageJ
        with open('imageJmacro.txt', 'a', encoding='utf-8') as mf:
            key = f'dset{i}'
            print(f"{key} >>> ...")
            n = images_16bit.shape[0]    
            print(">>>", n)
            for i in range(n):
                print(i)
                fn_in = f'{key}_frm{i}.tiff'
                fn_out = f'{key}_{i:03d}.csv'
                txt = (imagej_macro_template.format(PATH = PATH, fn_in = fn_in, fn_out = fn_out))
                mf.write(txt)
            mf.write('\n'*3)


