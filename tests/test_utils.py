from manify.utils.dataloaders import load


def test_dataloaders():
    print("Testing dataloaders")
    for dataset_name, dists_expected, labels_expected, adjacency_expected in [
        ("cities", True, False, False),
        ("cs_phds", True, True, True),
        # ("facebook", False, False, False),
        # ("power", False, False, False),
        ("polblogs", True, True, True),
        ("polbooks", True, True, True),
        ("cora", True, True, True),
        ("citeseer", True, True, True),
        # ("pubmed", True, True, True),
        ("karate_club", True, False, True),
        ("lesmis", True, False, True),
        ("adjnoun", True, False, True),
        ("football", True, False, True),
        ("dolphins", True, False, True),
        ("blood_cells", True, True, False),
        ("lymphoma", True, True, False),
        # ("cifar_100", True, True, False),
        # ("mnist", True, True, False),
        ("temperature", True, True, False),
        ("landmasses", True, True, False),
        ("neuron_33", True, True, False),
        ("neuron_46", True, True, False),
        ("traffic", True, True, False),
    ]:
        print(f"  Testing {dataset_name}")
        dists, labels, adjacency = load(dataset_name)
        if dists_expected:
            assert dists is not None, f"Distances should not be None for {dataset_name}"
        else:
            assert dists is None, f"Distances should be None for {dataset_name}"

        if labels_expected:
            assert labels is not None, f"Labels should not be None for {dataset_name}"
            if dists_expected:
                assert dists.shape[0] == labels.shape[0], "Number of labels should match number of distances"
        else:
            assert labels is None, f"Labels should be None for {dataset_name}"

        if adjacency_expected:
            assert adjacency is not None, f"Adjacency should not be None for {dataset_name}"
            if dists_expected:
                assert (
                    dists.shape[0] == adjacency.shape[0] == adjacency.shape[1]
                ), "Number of distances should match adjacency dimensions"
        else:
            assert adjacency is None, "Adjacency should be None for {dataset_name}"
        print("Done testing dataloaders")


if __name__ == "__main__":
    test_dataloaders()
