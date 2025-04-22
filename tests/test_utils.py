from manify.utils.dataloaders import load_hf


def test_dataloaders():
    print("Testing dataloaders")
    # I apologize for the handcoded nature of this, but this encodes my expectations for how each of the datasets
    # behaves. I added this to the docstring as well.
    for dataset_name, features_expected, dists_expected, labels_expected, adjacency_expected in [
        ("cities", False, True, False, False),
        ("cs_phds", False, True, True, True),
        ("polblogs", False, True, True, True),
        ("polbooks", False, True, True, True),
        ("cora", False, True, True, True),
        ("citeseer", False, True, True, True),
        ("karate_club", False, True, False, True),
        ("lesmis", False, True, False, True),
        ("adjnoun", False, True, False, True),
        ("football", False, True, False, True),
        ("dolphins", False, True, False, True),
        ("blood_cells", True, False, True, False),
        ("lymphoma", True, False, True, False),
        ("cifar_100", True, False, True, False),
        ("mnist", True, False, True, False),
        ("temperature", True, False, True, False),
        ("landmasses", True, False, True, False),
        ("neuron_33", True, False, True, False),
        ("neuron_46", True, False, True, False),
        ("traffic", True, False, True, False),
    ]:
        print(f"  Testing {dataset_name}")
        features, dists, adjacency, labels = load_hf(dataset_name)

        assert features_expected or dists_expected, "Must have features or distances"

        if features_expected:
            assert features is not None, f"Features should not be None for {dataset_name}"
            n = features.shape[0]
        else:
            assert features is None, f"Features should be None for {dataset_name}"

        if dists_expected:
            assert dists is not None, f"Distances should not be None for {dataset_name}"
            n = dists.shape[0]
        else:
            assert dists is None, f"Distances should be None for {dataset_name}"

        if adjacency_expected:
            assert adjacency is not None, f"Adjacency should not be None for {dataset_name}"
            assert adjacency.shape[0] == adjacency.shape[1] == n, "All adjacency matrix dimensions should be n"
        else:
            assert adjacency is None, "Adjacency should be None for {dataset_name}"

        if labels_expected:
            assert labels is not None, f"Labels should not be None for {dataset_name}"
            assert labels.shape[0] == n, "Number of labels should be n"
        else:
            assert labels is None, f"Labels should be None for {dataset_name}"

        print("Done testing dataloaders")
