## How to replicate the analysis

- navigate to `experimental-data`
- use `resample_images.ipynb.ipynb` to produce Monte Carlo samples
- navigate to /resampled
- run `process-resampled-cnn.py` and `process-resampled-rl.py` - OK, OK
- run `process-resampled-decode.py`
- finally, process the images using ThunderStorm:
  - run `prepare_for_thunderstorm.py` and then run the generated `imageJmacro.txt` in imageJ - OK
  - resulting localization data is then rendered into h5 files using `render_mef.py` - OK
- Then evaluate metrics with `evaluate-std-metrics-v2.ipynb`