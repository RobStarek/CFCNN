## How to replicate resolution test

* Use `generate_inputs.py` to produce testing data. Set MONTE_CARLO_SAMPLES = 10 to avoid multi-GB files. We enclose the export of results with MONTE_CARLO_SAMPLES = 100 for reference.
* Run `process_inputs_cnn.py` to transform testing data using CFCNN, resulting in `cnn_output_res_psf2hr.h5`
* Finally, use `process_cnn_outputs.ipynb` to evaluate the resolution. Processed results are saved in `results_psf2.h5` for later plotting.