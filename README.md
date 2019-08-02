# xcorr
Scripts for computing cross-correlations for SO

Requirements:

* `NaMaster`: [https://namaster.readthedocs.io/en/latest/pymaster.html](https://namaster.readthedocs.io/en/latest/pymaster.html)
* `pixell`: [https://github.com/simonsobs/pixell](https://github.com/simonsobs/pixell) (for flat sky computations)
* `healpy`: [https://healpy.readthedocs.io/en/latest](https://healpy.readthedocs.io/en/latest) (for curved sky computations)

How to run the Gaussian simulation code:

* From the xcorr directory, run: `python generate_mocks.py --path2config path/to/your/favorite/config.yaml`

How to compute Cls:

* From the xcorr directory, run: `python compute_cls.py --path2config path/to/your/favorite/config.yaml`
