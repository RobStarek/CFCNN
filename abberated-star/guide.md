## How to replicate the analysis of CFCNN tolerance to optical aberrations

* Run `generate_testing_data_single_v2.ipynb` to generate synthetic data with  single point.
* Run `generate_testing_data_star_v2.ipynb` to do the same for star-patterned data.
* Use `data/process_inputs_cnn.py`.
* To obtain the results, use `evaluate_abberations.ipynb`.