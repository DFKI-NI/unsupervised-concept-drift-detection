# Fully Unsupervised Concept Drift Detectors On Real-World Data Streams: An Empirical Study

## Abstract
Fully unsupervised concept drift detectors detected substantial changes in the patterns encoded in data streams by
observing the feature space only.
If left unaddressed, these changes could render other models deployed on the data stream unreliable.
This repository contains multiple fully unsupervised concept drift detectors, a framework to test various
configurations of these detectors on real-world data streams from the literature and the raw results from our 
experiments.

The implemented concept drift detectors are:
- Bayesian Non-parametric Detection Method (BNDM) [[doi]](https://doi.org/10.1145/3420034)
- Clustered Statistical test Drift Detection Method (CSDDM) [[url]](https://jit.ndhu.edu.tw/article/view/2504)
- Discriminative Drift Detector (D3) [[doi]](https://doi.org/10.1145/3357384.3358144)
- Ensemble Drift detection with Feature Subspaces (EDFS) [[doi]](https://doi.org/10.1109/DSAA.2019.00047)
- Image-Based Drift Detector (IBDD) [[doi]](https://doi.org/10.1109/BigData50022.2020.9377880)
- Nearest Neighbor-based Density Variation Identification (NN-DVI) [[doi]](https://doi.org/10.1016/j.patcog.2017.11.009)
- One-Class Drift Detector (OCDD) [[doi]](https://doi.org/10.1145/3357384.3358144)
- Semi-Parametric Log Likelihood (SPLL) [[doi]](https://doi.org/10.1109/TKDE.2011.226)
- Unsupervised Concept Drift Detection (UCDD) [[doi]](https://doi.org/10.1142/9789811223334_0017)
- Unsupervised Change Detection for Activity Recognition (UDetect) [[doi]](https://doi.org/10.1108/IJPCC-03-2017-0027)

Experiment results show that D3 and SPLL are the best-performing detectors across various data streams and metrics.
Each detector's corresponding publication is listed in the respective detector's file, found in the folder `detectors`.

If you use components from this repository, please cite this work as:
```
TODO
```
If you use a concept drift detector, please cite the corresponding publication of the respective authors as well.

## Install
To install the required dependencies and data sets, follow these steps.
This study is implemented in `Python 3.8`.
If you do not intend to reproduce any experiments, you may skip steps 2 and 3.
1. Install dependencies, e.g., `pip install -r requirements.txt`
2. Download the data sets from the [USP DS Repository](https://sites.google.com/view/uspdsrepository) and extract them in `datasets/files`. Note that the archive is encrypted. Souza et al. provide the password in the corresponding publication titled _Challenges in Benchmarking Stream Learning Algorithms with Real-world Data_ [[doi]](https://doi.org/10.1007/s10618-020-00698-5).
3. Verify that the data sets are located in `datasets/files`, e.g., `datasets/files/outdoor.arff`.
4. Execute `python convert_datasets.py` to convert the data sets to CSV and convert the class labels to pandas-readable characters.
5. Test by executing `python -m unittest discover -s test -t .`.

## Execute
If you wish to reproduce our experiments, follow all installation steps above. 
Then, execute `python main.py <your_experiment_name>`. 
The results will be saved in `results/<data stream>/<detector>_<your_experiment_name>.csv`.
You may provide the number of threads to use by setting `OMP_NUM_THREADS`: `OMP_NUM_THREADS=8 python main.py full-test`.
`config.py` contains the full configuration used in our experiments.
Note that repeating all experiments may take several months, depending on your hardware.

If you want to create the results data the figures and tables are based on, execute `python eval.py`.
Several directories labeled `results_best`, `results_summarized` etc. will be created and filled with data.
They provide the following content in order of creation by `eval.py`:
- `results` contains the raw experiment logs (not created during evaluation)
- `results_no_detections` contains all configurations that failed to detect any concept drift
- `results_periodic` contains all configurations that detected periodic, i.e., every _n_ time steps
- `results_figures` contains all figures as .eps files. If you wish to view the figures directly, set `show=True` at the top of `eval.py`.
- `results_clean` contains filtered experiment results, that contain no lines featuring periodic detection or no detection at all
- `results_summarized` contains aggregated experiment results containing the mean and std of all recorded metrics (based on clean results)
- `results_best` contains the peak results in terms of accuracy and lift-per-drift for each detector (based on summarized results)

## Funding
This study was conducted in the project _Change Event based Sensor Sampling (ChESS)_ at the department for Marine Perception
at the German Research Center for Artificial Intelligence. ChESS was funded by the Volkswagen Stiftung (Volkswagen
Foundation) and Niedersächsisches Ministerium für Wissenschaft und Kultur (Lower Saxony's Ministry for Science and
Culture) under Niedersächsisches Vorab (grant number ZN3683).