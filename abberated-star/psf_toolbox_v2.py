"""
psf_toolbox.py
This module provides tools for generating point spread functions (PSFs) and rendering emitter distributions convolved with a PSF.

Functions:
    - bin_image: Downsamples a grayscale image by binning blocks and replacing them with their mean.
    - psf_high_NA_2D: Generates a 2D in-focus scalar PSF with Zernike aberrations and √cosθ apodisation.
    - scaled_psf: Returns a 50x50 PSF, normalized, with fixed pupil sampling and padding.
    - scaled_psf2: Returns a 50x50 PSF for arbitrary pixel size in Airy units, using fast interpolation.
    - render_psf_emitters: Renders a set of emitters convolved with a high-resolution PSF and rescales to 50x50.

Dependencies:
    numpy, cv2, matplotlib, scipy.ndimage, h5py


Examples
--------
See notebooks, like generate_testing_data_star_v2.ipynb
"""

import numpy as np
import cv2
from numpy.fft import fftshift, ifft2, ifftshift
import scipy.ndimage as ndi
import h5py


def bin_image(img: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Downsamples a grayscale image by binning. Each bin_size x bin_size block
    is replaced with its mean, reducing the image size.

    Parameters:
        img (np.ndarray): 2D input image (dtype=uint8).
        bin_size (int): Binning factor (block size).

    Returns:
        np.ndarray: Downsampled image (dtype=uint8).
    """
    h, w = img.shape
    h_trim = h - h % bin_size
    w_trim = w - w % bin_size

    img_trimmed = img[:h_trim, :w_trim]

    # Reshape into 4D: (new_h, bin_size, new_w, bin_size)
    reshaped = img_trimmed.reshape(
        h_trim // bin_size, bin_size, w_trim // bin_size, bin_size
    )

    # Compute mean over binning dimensions
    binned = reshaped.mean(axis=(1, 3))

    # Round and convert to uint8
    return binned

def psf_high_NA_vector(N_pupil=256, pad_factor=8,
                       wavelength=0.580,        # µm
                       NA=1.40, n=1.5,
                       pol='circ',
                       A_def=0.0, A_ast=0.0, A_ast_y=0.0,
                       A_coma_x=0.0, A_coma_y=0.0,
                       A_trefoil_x=0.0, A_trefoil_y=0.0, A_sph=0.0):
    """
    Vectorial in-focus PSF via Debye–Wolf integral with Zernike aberrations.

    Parameters
    ----------
    N_pupil    : samples across aperture diameter
    pad_factor : zero-padding factor for image plane sampling
    wavelength : wavelength [µm]
    NA, n      : numerical aperture, immersion index
    pol        : 'x', 'y', or 'lin45' input polarization
    Aberration coeffs in waves at the pupil edge.
    
    Returns
    -------
    psf : (M,M) normalised intensity
    x,y : coordinates in µm
    """
    k = 2 * np.pi / wavelength
    M  = N_pupil * pad_factor
    P  = np.zeros((M, M), dtype=complex)

    mid    = M // 2
    half   = N_pupil // 2
    sl     = slice(mid - half, mid + half)
    u,v    = np.mgrid[-1:1:N_pupil*1j, -1:1:N_pupil*1j]
    rho    = np.sqrt(u**2 + v**2)
    theta  = np.arctan2(v, u)
    mask   = rho <= 1

    # Map pupil radius rho -> sinθ
    sin_t_max = NA / n
    sin_t     = rho * sin_t_max
    sin_t[~mask] = 0.0
    cos_t     = np.sqrt(1 - sin_t**2)

    # Zernike polynomials (unnormalised)
    Z20 = 2*rho**2 - 1
    Z22 = rho**2 * np.cos(2*theta)
    Z2m2 = rho**2 * np.sin(2*theta)
    Z31 = (3*rho**3 - 2*rho) * np.cos(theta)
    Z3m1 = (3*rho**3 - 2*rho) * np.sin(theta)
    Z33 = rho**3 * np.cos(3*theta)
    Z3m3 = rho**3 * np.sin(3*theta)
    Z40 = 6*rho**4 - 6*rho**2 + 1

    # Normalize Zernikes over pupil
    for Z in (Z20, Z22, Z2m2, Z31, Z3m1, Z33, Z3m3, Z40):
        Z /= np.sqrt(np.mean(Z[mask]**2))

    phase = 2 * np.pi * (
        A_def       * Z20 +
        A_ast       * Z22 +
        A_ast_y     * Z2m2 +
        A_coma_x    * Z31 +
        A_coma_y    * Z3m1 +
        A_trefoil_x * Z33 +
        A_trefoil_y * Z3m3 +
        A_sph       * Z40
    )

    # Vectorial pupil field (Richards–Wolf)
    if pol.lower() == 'x':
        Ex_pupil = cos_t + (1 - cos_t) * np.cos(2*theta)
        Ey_pupil = (1 - cos_t) * np.sin(2*theta)
    elif pol.lower() == 'y':
        Ex_pupil = (1 - cos_t) * np.sin(2*theta)
        Ey_pupil = cos_t - (1 - cos_t) * np.cos(2*theta)
    elif pol.lower() == 'lin45':
        Ex_lin = cos_t + (1 - cos_t) * np.cos(2*theta)
        Ey_lin = (1 - cos_t) * np.sin(2*theta)
        Ex_pupil = (Ex_lin + Ey_lin) / np.sqrt(2)
        Ey_pupil = (Ey_lin + (cos_t - (1 - cos_t) * np.cos(2*theta))) / np.sqrt(2)
    elif pol.lower() == 'circ':
        # Circular polarization = (x-pol + i y-pol) / sqrt(2)
        Ex_x = cos_t + (1 - cos_t) * np.cos(2*theta)
        Ey_x = (1 - cos_t) * np.sin(2*theta)
        Ex_y = (1 - cos_t) * np.sin(2*theta)
        Ey_y = cos_t - (1 - cos_t) * np.cos(2*theta)
        Ex_pupil = (Ex_x + 1j * Ex_y) / np.sqrt(2)
        Ey_pupil = (Ey_x + 1j * Ey_y) / np.sqrt(2)
    else:
        raise ValueError("pol must be 'x', 'y', 'lin45', or 'circ'")

    # Apodisation = sqrt(cosθ) from Debye–Wolf theory
    apod = np.sqrt(cos_t) * mask

    # Apply phase and apodisation
    Px = apod * Ex_pupil * np.exp(1j*phase)
    Py = apod * Ey_pupil * np.exp(1j*phase)

    # Insert into padded array
    Px_full = np.zeros_like(P)
    Py_full = np.zeros_like(P)
    Px_full[sl, sl] = Px
    Py_full[sl, sl] = Py

    # FFT to image plane
    fx = fftshift(ifft2(ifftshift(Px_full)))
    fy = fftshift(ifft2(ifftshift(Py_full)))

    # Intensity from vector components
    psf = np.abs(fx)**2 + np.abs(fy)**2
    psf /= psf.sum()

    # Coordinates in µm
    df = (NA / wavelength) / (N_pupil / 2)   # cycles/µm
    dx = 1 / (M * df)                        # µm per pixel
    coords = (np.arange(M) - mid) * dx

    return psf, coords, coords

def render_psf_emitters(
    xyi: np.ndarray,
    px_size_au: float,
    psf_hires: np.ndarray,
    psf_hires_dx: float,
    h: int = 50,
    w: int = 50,
) -> np.ndarray:
    """
    Renders a set of emitters convolved with a high-resolution PSF and rescales to a 50x50 output using interpolation.

    Parameters:
        xyi (np.ndarray): Array of emitters, each row as (x, y, intensity) in pixel coordinates.
        px_size_au (float): Pixel size in Airy units for the output image.
        psf_hires (np.ndarray): High-resolution PSF array for convolution. Be sure to crop it to necessary size (e.g.50px) to avoid memory errors.
        psf_hires_dx (float): Pixel size of the high-resolution PSF in Airy units.
        h (int, optional): Output image height (default 50).
        w (int, optional): Output image width (default 50).
    Returns:
        np.ndarray: 50x50 normalized image of emitters convolved with the PSF.
    """
    scale = px_size_au / psf_hires_dx
    hires_size = int(round(h * scale))
    imgh = np.zeros((hires_size, hires_size))
    OFFSET = 0.5
    for x, y, i in xyi:
        xup = int(round(scale * x +OFFSET))
        yup = int(round(scale * y +OFFSET))
        if xup < 0 or xup >= hires_size:
            continue
        if yup < 0 or yup >= hires_size:
            continue
        imgh[yup, xup] += i
    imghr = ndi.convolve(imgh, psf_hires)
    midh = int(hires_size // 2)
    midv = int(hires_size // 2)
    hw = int(scale * w // 2)
    hh = int(scale * h // 2)
    cropped = imghr[midh - hh : midh + hh, midv - hw : midv + hw]
    scaled = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_CUBIC)
    # scaled = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_LINEAR)
    return scaled
