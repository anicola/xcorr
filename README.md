# xcorr
Scripts for computing cross-correlations for SO

Requirements:

* Install `NaMaster`: [https://namaster.readthedocs.io/en/latest/pymaster.html](https://namaster.readthedocs.io/en/latest/pymaster.html)
* Install `pixell`: [https://github.com/simonsobs/pixell](https://github.com/simonsobs/pixell)

How to run the Gaussian simulation code:

* From the xcorr directory, run: `python generate_mocks.py --path2config path/to/your/favorite/config.yaml`

How to compute Cls:

* From the xcorr directory, run: `python compute_cls.py --path2config path/to/your/favorite/config.yaml`
