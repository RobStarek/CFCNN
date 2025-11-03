## How to replicate the `star-pattern` analysis

* Run `gen_inputs.ipynb`.
* Run `process_inputs_cnn.py`.
* Run `rl_process.py`.
* Run `process_decode_individual.ipynb` to produce DECODE-reconstructed images.
* Obtain MEF results:
  * Run `convert_inputs_for_mef.py`.
  * Process the TIFF file using ImageJ/ThunderSTORM with the generated `MEFImageJmacro.txt`.
  * Render localized emitters using `render_mef.py`.
* Use the notebook `generate_figure.ipynb` to produce the final results.