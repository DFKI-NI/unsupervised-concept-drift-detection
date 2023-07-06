from datasets import (
    InsectsAbruptBalanced,
    InsectsAbruptImbalanced,
    InsectsGradualBalanced,
    InsectsGradualImbalanced,
    InsectsIncrementalAbruptBalanced,
    InsectsIncrementalAbruptImbalanced,
    InsectsIncrementalBalanced,
    InsectsIncrementalImbalanced,
    InsectsIncrementalReoccurringBalanced,
    InsectsIncrementalReoccurringImbalanced,
)
from detectors import *
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
    streams = [
        InsectsAbruptBalanced(),
        InsectsAbruptImbalanced(),
        InsectsGradualBalanced(),
        InsectsGradualImbalanced(),
        InsectsIncrementalAbruptBalanced(),
        InsectsIncrementalAbruptImbalanced(),
        InsectsIncrementalBalanced(),
        InsectsIncrementalImbalanced(),
        InsectsIncrementalReoccurringBalanced(),
        InsectsIncrementalReoccurringImbalanced(),
    ]
    n_training_samples = 1000
    models = [
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[100, 250, 500, 1000, 2000]),
                Parameter("recent_samples_proportion", values=[0.1, 0.3, 0.5, 1.0]),
                Parameter("threshold", values=[0.6, 0.7, 0.8]),
            ],
            seeds=None,
            n_runs=10,
        ),
        ModelOptimizer(
            base_model=OneClassDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000, 2000]),
                Parameter("threshold", values=[0.2, 0.3, 0.4, 0.5]),
                Parameter("outlier_detector_kwargs", value={"nu": 0.5, "kernel": "rbf", "gamma": "auto"})
            ],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", values=[25, 50, 100]),
                Parameter("n_samples", values=[50, 100, 250, 500]),
                Parameter("disjoint_training_windows", value=True)
            ],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", values=[50, 100, 250]),
                Parameter("n_samples", values=[100, 250, 500, 1000, 2000]),
                Parameter("disjoint_training_windows", value=False)
            ],
            seeds=None,
            n_runs=1,
        )
    ]
