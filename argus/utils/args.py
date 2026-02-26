import argparse
from loguru import logger

def generate_svgp_args(**kwargs):
    if 'inducing_points' not in kwargs.keys():
        raise ValueError("You need to provide some inducing points from the input space.")
    
    parser = argparse.ArgumentParser(description="Args for SVGP model")
    
    import torch
    predefined_args = {
        "num_concepts": 4,
        "lengthscale":0.1,
        "outputscale":1.0,
        "n_inducing_points": 25,
        "gp_training_iter": 300,
        "inputnoise":0.1,
        "num_likelihood_samples": 50,
        "learnnoise": True,
        "learn_mean": True,
        "learnoutputscale": True,
        "learnlengthscale": True,
        "seed":42,
        "lr": 0.01,
        "device": 'cuda',
    }
    # Add predefined args with defaults, overridden by kwargs if present
    for key, default in predefined_args.items():
        value = kwargs.get(key, default)
        logger.debug(f"Adding {key}:{value}")
        parser.add_argument(f"-{key}", type=type(value), default=value)

    # Add any extra kwargs not already added
    for key, value in kwargs.items():
        logger.debug(f"Adding {key}:{value}")
        if key not in predefined_args:
            parser.add_argument(f"-{key}", type=type(value), default=value)
            
    args = parser.parse_args([])  # Empty list because no command-line input
    return args  # You can also return vars(args) to get a dict


def generate_mcsvgp_args(**kwargs):
    if 'inducing_points' not in kwargs.keys():
        raise ValueError("You need to provide some inducing points from the input space.")
    
    parser = argparse.ArgumentParser(description="Args for SVGP model")
    
    import torch
    predefined_args = {
        "num_concepts": 4,
        "lengthscale":0.1,
        "outputscale":1.0,
        "n_inducing_points": 25,
        "gp_training_iter": 300,
        "inputnoise":0.1,
        "num_likelihood_samples": 50,
        "learnnoise": True,
        "learn_mean": True,
        "learnoutputscale": True,
        "learnlengthscale": True,
        "seed":42,
        "lr": 0.01,
        "device": 'cuda',
    }
    # Add predefined args with defaults, overridden by kwargs if present
    for key, default in predefined_args.items():
        value = kwargs.get(key, default)
        logger.debug(f"Adding {key}:{value}")
        parser.add_argument(f"-{key}", type=type(value), default=value)

    # Add any extra kwargs not already added
    for key, value in kwargs.items():
        logger.debug(f"Adding {key}:{value}")
        if key not in predefined_args:
            parser.add_argument(f"-{key}", type=type(value), default=value)
            
    args = parser.parse_args([])  # Empty list because no command-line input
    return args  # You can also return vars(args) to get a dict


def generate_ysvgp_args(**kwargs):
    if 'inducing_points' not in kwargs.keys():
        raise ValueError("You need to provide some inducing points from the input space.")
    
    parser = argparse.ArgumentParser(description="Args for SVGP model")
    
    import torch
    predefined_args = {
        "num_concepts": 4,
        "lengthscale":0.1,
        "outputscale":1.0,
        "n_inducing_points": 25,
        "gp_training_iter": 300,
        "inputnoise":0.1,
        "num_likelihood_samples": 50,
        "learnnoise": True,
        "learn_mean": True,
        "learnoutputscale": True,
        "learnlengthscale": True,
        "seed":42,
        "lr": 0.01,
        "device": 'cuda',
    }
    # Add predefined args with defaults, overridden by kwargs if present
    for key, default in predefined_args.items():
        value = kwargs.get(key, default)
        logger.debug(f"Adding {key}:{value}")
        parser.add_argument(f"-{key}", type=type(value), default=value)

    # Add any extra kwargs not already added
    for key, value in kwargs.items():
        logger.debug(f"Adding {key}:{value}")
        if key not in predefined_args:
            parser.add_argument(f"-{key}", type=type(value), default=value)
            
    args = parser.parse_args([])  # Empty list because no command-line input
    return args  # You can also return vars(args) to get a dict