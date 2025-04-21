# Lazy loading variant of utils imports - changed to this so that missing imports would be non-breaking for core manify


def __getattr__(name):
    if name == "benchmarks":
        import manify.utils.benchmarks

        return manify.utils.benchmarks
    elif name == "dataloaders":
        import manify.utils.dataloaders

        return manify.utils.dataloaders
    elif name == "link_prediction":
        import manify.utils.link_prediction

        return manify.utils.link_prediction
    elif name == "preprocessing":
        import manify.utils.preprocessing

        return manify.utils.preprocessing
    elif name == "visualization":
        import manify.utils.visualization

        return manify.utils.visualization
    else:
        raise AttributeError(f"module 'manify.utils' has no attribute '{name}'")
