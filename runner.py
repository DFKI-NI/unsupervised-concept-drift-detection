from config import Configuration


def run(experiment_name):
    for stream in Configuration.streams:
        for model in Configuration.models:
            model.optimize(stream, experiment_name, Configuration.n_training_samples, verbose=True)
