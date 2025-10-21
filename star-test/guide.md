## How to replicate star-pattern analysis

* Run `generate_inputs.py`
* Run `process_inputs_cnn.py`.
* Run `rl_process.py`.
* Use `process_decode_individual.ipynb` to produce DECODE-reconstructed images.
* Obtain MEF results:
  * Run `convert_inputs_for_mef.py`
  * Process tiff file using imageJ/ThunderStorm, with the help of generated `MEFImageJmacro.txt`.
  * Render localized emitters using `render_mef.py`.
* Get the final results with notebook `generate_figure.ipynb`.