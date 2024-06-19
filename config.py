from datasets import (
    Electricity,
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalAbruptBalanced,
    InsectsIncrementalBalanced,
    InsectsIncrementalReoccurringBalanced,
    NOAAWeather,
    OutdoorObjects,
    PokerHand,
    Powersupply,
    RialtoBridgeTimelapse,
    SineClusters,
    WaveformDrift2
)
from detectors import *
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
    streams = [
        Electricity(),
        InsectsAbruptBalanced(),
        InsectsGradualBalanced(),
        InsectsIncrementalAbruptBalanced(),
        InsectsIncrementalBalanced(),
        InsectsIncrementalReoccurringBalanced(),
        NOAAWeather(),
        OutdoorObjects(),
        PokerHand(),
        Powersupply(),
        RialtoBridgeTimelapse(),
        SineClusters(drift_frequency=5000, stream_length=154_987, seed=531874),
        WaveformDrift2(drift_frequency=5000, stream_length=154_987, seed=2401137),
    ]
    n_training_samples = 1000
    models = [
        ModelOptimizer(
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("const", values=[0.5, 1.0]),
                Parameter("max_depth", values=[2, 3]),
                Parameter("threshold", values=[0.45, 0.5, 0.55]),
            ],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=ClusteredStatisticalTestDriftDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("confidence", values=[0.1, 0.01]),
                Parameter("feature_proportion", values=[0.1, 0.01]),
                Parameter("n_clusters", values=[2, 3]),
            ],
            seeds=None,
            n_runs=5,
        ),
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[50, 125, 250, 500]),
                Parameter("recent_samples_proportion", values=[0.1, 0.5, 1.0]),
                Parameter("threshold", values=[0.6, 0.7, 0.8]),
            ],
            seeds=None,
            n_runs=5,
        ),
        ModelOptimizer(
            base_model=ImageBasedDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("n_permutations", values=[10, 20, 40]),
                Parameter("update_interval", values=[50, 100, 250]),
                Parameter("n_consecutive_deviations", values=[1, 4]),
            ],
            seeds=None,
            n_runs=5,
        ),
        ModelOptimizer(
            base_model=OneClassDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("threshold", values=[0.2, 0.3, 0.4, 0.5]),
                Parameter("outlier_detector_kwargs", value={"nu": 0.5, "kernel": "rbf", "gamma": "auto"})
            ],
            seeds=None,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=SemiParametricLogLikelihood,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("n_clusters", values=[2, 3]),
                Parameter("threshold", values=[0.05, 0.005]),
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
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("disjoint_training_windows", value=False)
            ],
            seeds=None,
            n_runs=1,
        )
    ]
