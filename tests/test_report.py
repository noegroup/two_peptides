
from two_peptides.run import run
from torchmdnet.datasets.in_mem_dataset import InMemoryDataset


def test_run_and_read(tmpdir):
    run("A", "A", str(tmpdir), test=True)
    InMemoryDataset(
        coordglob=str(tmpdir/"test_*_coord.npy"),
        forceglob=str(tmpdir/"test_*_force.npy"),
        embedglob=str(tmpdir/"test_*_embed.npy"),
    )
