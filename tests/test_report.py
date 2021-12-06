
import numpy as np
from two_peptides.run import run, DEFAULT_DISTANCES
from torchmdnet.datasets.in_mem_dataset import InMemoryDataset


def test_run_and_read(tmpdir):
    run("A", "A", str(tmpdir), test=True)
    dataset = InMemoryDataset(
        coordglob=str(tmpdir/"test_*_coord.npy"),
        forceglob=str(tmpdir/"test_*_force.npy"),
        embedglob=str(tmpdir/"test_*_embed.npy"),
    )
    assert dataset[0].pos.shape == (10, 3)
    assert dataset[0].y.shape == (10, 3)
    assert dataset[0].z.shape == (10, )
    assert len(dataset) == len(DEFAULT_DISTANCES) * 10
