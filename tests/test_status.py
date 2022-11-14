

from two_peptides.status import submitted


def test_status():
    assert len(list(submitted())) > 900

