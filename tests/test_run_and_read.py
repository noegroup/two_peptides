

from torch_geometric.data import DataLoader
from two_peptides.run import run
from two_peptides.meta import DEFAULT_DISTANCES
from two_peptides.pmf import TwoPeptidesDataset


def test_run_and_read(tmpdir):
    run("A", "A", str(tmpdir), test=True)
    run("A", "Y", str(tmpdir), test=True)
    run("Y", "Y", str(tmpdir), test=True)
    dataset = TwoPeptidesDataset(
        coordglob=str(tmpdir/"test_*_coord.npy"),
        forceglob=str(tmpdir/"test_*_force.npy"),
        embedglob=str(tmpdir/"test_*_embed.npy"),
    )
    assert dataset[0].pos.shape == (44, 3)
    assert dataset[0].y.shape == (44, 3)
    assert dataset[0].z.shape == (44, )
    assert len(dataset) == len(DEFAULT_DISTANCES) * 10 * 3

    # test split
    trainset, valset = dataset.split(0.333)
    assert len(trainset) == len(DEFAULT_DISTANCES) * 10 * 2
    assert len(valset) == len(DEFAULT_DISTANCES) * 10

    assert len(trainset + valset) == len(DEFAULT_DISTANCES) * 10 * 3
    trainloader = DataLoader(trainset, shuffle=True, batch_size=256)
    for batch in trainloader:
        # make sure unit conversion has been applied
        assert batch.pos.std() > 3.0
