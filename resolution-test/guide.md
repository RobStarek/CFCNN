## How to replicate the resolution test

* Run `generate_inputs.py` to produce test data. Set `MONTE_CARLO_SAMPLES = 10` to avoid multi-GB files. The repository includes results produced with `MONTE_CARLO_SAMPLES = 100` for reference.
* Run `process_inputs_cnn.py` to process the test data with CFCNN; this creates `cnn_output_res_psf2hr.h5`.
* Open `process_cnn_outputs.ipynb` to evaluate resolution. The processed results are saved to `results_psf2.h5` for plotting.