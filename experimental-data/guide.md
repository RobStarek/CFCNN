## How to replicate the analysis

- navigate to `experimental-data`
- use `resample_images.ipynb` to produce Monte Carlo samples
- process data with DECODE, use `decode_render.ipynb` and subsequently `add_decode_to_h5.py`.
- navigate to /resampled
- run `process-resampled-cnn.py` and `process-resampled-rl.py` - OK, OK
- run `process-resampled-decode.py`
- finally, process the images using ThunderStorm:
  - run `prepare_for_thunderstorm.py` and then run the generated `imageJmacro.txt` in imageJ - OK
  - resulting localization data is then rendered into h5 files using `render_mef.py` - OK
- process resampled data with DECODE, use `process-resampled-decode.py`
- evaluate metrics with `evaluate-std-metrics-v2.ipynb` to obtain metrics table with uncertainties
- use `Figure2.ipynb` to produce the figure in the manuscript

