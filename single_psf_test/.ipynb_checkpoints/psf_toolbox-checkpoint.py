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
```
#Definitions
import matplotlib.pyplot as plt
au = 580*0.61/1.4 # airy unit
##Example 1:
psf2 = scaled_psf2(58/au, 1.4, 1.5, A_def=0.05)
plt.matshow(psf2)
##Example 2:
psfhr, x, y = psf_high_NA_2D(256, 8, A_def = 0.05)
psfhr = psfhr[1024-64 : 1024+64, 1024-64 : 1024+64]
xyi = np.array([
    (10.4,10,1),
    (10.45,20,1),
    (25,25,1),
])
au = 580*0.61/1.4 #airy unit
im = render_psf_emitters(xyi, 59/au, psfhr, 0.125, 50, 50)
plt.matshow(im)
```

"""

import numpy as np
import cv2
from numpy.fft import fftshift, ifft2, ifftshift
# import matplotlib.pyplot as plt
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


def psf_high_NA_2D(
    N_pupil: int = 256,
    pad_factor: int = 8,
    A_def: float = 0.0,
    A_ast: float = 0.0,
    A_ast_y: float = 0.0,
    A_coma_x: float = 0.0,
    A_coma_y: float = 0.0,
    A_trefoil_x: float = 0.0,
    A_trefoil_y: float = 0.0,
    A_sph: float = 0.0,
    NA: float = 1.40,
    n: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    In‑focus scalar PSF (√cosθ apodisation) with Zernike aberrations.

    Parameters
    ----------
    N_pupil    : int
        Samples across the aperture diameter.
    pad_factor : int
        Total grid = N_pupil * pad_factor (controls image sampling).
    A_def      : float
        Defocus aberration [waves].
    A_ast      : float
        Astigmatism (0°) [waves].
    A_ast_y    : float
        Astigmatism (90°) [waves].
    A_coma_x   : float
        Coma (x) [waves].
    A_coma_y   : float
        Coma (y) [waves].
    A_trefoil_x: float
        Trefoil (x) [waves].
    A_trefoil_y: float
        Trefoil (y) [waves].
    A_sph      : float
        Spherical aberration [waves].
    NA         : float
        Objective NA.
    n          : float
        Immersion refractive index.

    Returns
    -------
    psf : np.ndarray
        (M, M) normalised intensity.
    x   : np.ndarray
        Coordinate vector in Airy‑radius units (1 ≈ 0.61 λ/NA).
    y   : np.ndarray
        Coordinate vector in Airy‑radius units (1 ≈ 0.61 λ/NA).
    """
    M = N_pupil * pad_factor
    P = np.zeros((M, M), dtype=complex)

    # ── build a centred N_pupil × N_pupil pupil tile ────────────────────
    mid = M // 2
    half = N_pupil // 2
    sl = slice(mid - half, mid + half)  # indices of the pupil block
    u, v = np.mgrid[-1 : 1 : N_pupil * 1j, -1 : 1 : N_pupil * 1j]
    rho = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)
    mask = rho <= 1

    # Zernikes (unnormalised)
    Z20 = 2 * rho**2 - 1
    Z22 = rho**2 * np.cos(2 * theta)
    Z2m2 = rho**2 * np.sin(2 * theta)
    Z31 = (3 * rho**3 - 2 * rho) * np.cos(theta)
    Z3m1 = (3 * rho**3 - 2 * rho) * np.sin(theta)
    Z33 = rho**3 * np.cos(3 * theta)
    Z3m3 = rho**3 * np.sin(3 * theta)
    Z40 = 6 * rho**4 - 6 * rho**2 + 1

    # Zernikes (normalised)
    Z20 /= np.sqrt(np.mean(Z20[mask] ** 2))
    Z22 /= np.sqrt(np.mean(Z22[mask] ** 2))
    Z2m2 /= np.sqrt(np.mean(Z2m2[mask] ** 2))
    Z31 /= np.sqrt(np.mean(Z31[mask] ** 2))
    Z3m1 /= np.sqrt(np.mean(Z3m1[mask] ** 2))
    Z33 /= np.sqrt(np.mean(Z33[mask] ** 2))
    Z3m3 /= np.sqrt(np.mean(Z3m3[mask] ** 2))
    Z40 /= np.sqrt(np.mean(Z40[mask] ** 2))

    phase = (
        2
        * np.pi
        * (
            A_def * Z20
            + A_ast * Z22
            + A_ast_y * Z2m2
            + A_coma_x * Z31
            + A_coma_y * Z3m1
            + A_trefoil_x * Z33
            + A_trefoil_y * Z3m3
            + A_sph * Z40
        )
    )

    # √cosθ apodisation with clipping
    sin_t_max = NA / n
    sin_t = np.clip(rho * sin_t_max, 0, 1)
    apod = np.sqrt(np.sqrt(1 - sin_t**2)) * mask

    P[sl, sl] = apod * np.exp(1j * phase)  # write into padded grid

    # ── FFT to image plane ──────────────────────────────────────────────
    field = fftshift(ifft2(ifftshift(P)))
    psf = np.abs(field) ** 2
    psf /= psf.sum()
    # psf  /= psf.max()
    # coordinate axes in Airy‑radius units
    # 1 pixel ≈ 1/pad_factor Airy radii
    coords = (np.arange(M) - mid) / pad_factor
    return psf, coords, coords


def scaled_psf(NA: float = 1.4, n: float = 1.5, **A_kwargs) -> np.ndarray:
    """
    Generates a scaled 2D point spread function (PSF) array with specified numerical aperture (NA) and refractive index (n).
    It matched our ~60 nm px by some approximation.

    The function computes a high-resolution PSF using fixed pupil sampling (256) and padding factor (8), then bins and crops the result to a 50x50 region centered on the PSF. It normalizes the cropped PSF so that its sum is 1.
    If the PSF extends beyond the cropped region (i.e., significant intensity >1% of maximmum at the edges), a warning is printed.
    Parameters:
        NA (float, optional): Numerical aperture of the objective. Default is 1.4.
        n (float, optional): Refractive index of the medium. Default is 1.5.
        **A_kwargs: Additional keyword arguments passed to `psf_high_NA_2D`.
    Returns:
        np.ndarray: A 50x50 normalized PSF array.
    Warns:
        Prints a warning if the PSF is likely larger than the 50x50 cropped region.
    """
    # if A_kwargs is None:
    #     A_kwargs = dict()
    hires, x, y = psf_high_NA_2D(256, 8, NA=NA, n=n, **A_kwargs)
    binned = bin_image(hires, 2)
    mid = 512  # binned center
    cropped = binned[mid - 25 : mid + 25, mid - 25 : mid + 25]
    mxm = np.max(cropped)

    # test whether we do not crop too much, tolerate 1% of maximum, otherwise print warning:
    try:
        assert cropped[0, 0] / mxm < 1e-2
    except AssertionError:
        print("Warning: PSF is likely larger than 50x50")

    return cropped / np.sum(cropped)


def scaled_psf2(px_size_au, NA: float = 1.4, n: float = 1.5, **A_kwargs) -> np.ndarray:
    """
    Generates a scaled 2D point spread function (PSF) array with specified pixel size and optical parameters.

    This function computes a high-resolution PSF using the `psf_high_NA_2D` function, crops it to a region corresponding to the desired physical size, and rescales it to a 50x50 array using cubic interpolation. The resulting PSF is normalized to sum to 1.

    Parameters:
        px_size_au (float): Pixel size in Airy units.
        NA (float, optional): Numerical aperture of the objective. Default is 1.4.
        n (float, optional): Refractive index of the medium. Default is 1.5.
        **A_kwargs: Additional keyword arguments passed to `psf_high_NA_2D`.

    Returns:
        np.ndarray: A 50x50 normalized PSF array.
    """
    hires, x, y = psf_high_NA_2D(256, 8, NA=NA, n=n, **A_kwargs)
    dx = x[1] - x[0]
    halfwidth_dx = 25 * px_size_au
    mid = 1024
    halfwidth_dx_steps = int(round(halfwidth_dx / dx))
    cropped = hires[
        mid - halfwidth_dx_steps : mid + halfwidth_dx_steps,
        mid - halfwidth_dx_steps : mid + halfwidth_dx_steps,
    ]
    scaled = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_CUBIC)
    return scaled / np.sum(scaled)


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
    for x, y, i in xyi:
        xup = int(round(scale * x +0.5))
        yup = int(round(scale * y +0.5))
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
    return scaled


# ##Example 1:
# psf = scaled_psf(1.4, 1.5, A_def=0.05)
# plt.matshow(psf)
# plt.show()

# au = 580*0.61/1.4 #airy unit
# psf2 = scaled_psf2(58/au,1.4, 1.5, A_def=0.05)
# plt.matshow(psf2)
# plt.show()

# ##Example 2:
# psfhr, x, y = psf_high_NA_2D(256, 8, A_def = 0.05)
# psfhr = psfhr[1024-64 : 1024+64, 1024-64 : 1024+64]
# xyi = np.array([
#     (10.4,10,1),
#     (10.45,20,1),
#     (25,25,1),
# ])
# au = 580*0.61/1.4 #airy unit
# im = render_psf_emitters(xyi, 59/au, psfhr, 0.125, 50, 50)
# plt.matshow(im)
