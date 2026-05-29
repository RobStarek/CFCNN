## How to replicate the analysis

- Navigate to `experimental-data`.
- Use `resample_images.ipynb` to produce Monte Carlo samples.
- Process data with DECODE using `decode_render.ipynb` and subsequently `add_decode_to_h5.py`.
- Navigate to `/resampled`.
- Run `process-resampled-cnn.py` and `process-resampled-rl.py`.
- Run `process-resampled-decode.py`.
- Process the images using ThunderSTORM:
  - Run `prepare_for_thunderstorm.py` and then run the generated `imageJmacro.txt` in ImageJ.
  - Render localization data into HDF5 files using `render_mef.py`.
- Evaluate metrics with `evaluate-std-metrics-v3.ipynb` to obtain metrics table with uncertainties.
- Use `Figure2.ipynb` to produce the figure in the manuscript.

