

from two_peptides.meta import fast_folder_pairs


def test_fast_folder_pairs():
    assert len(fast_folder_pairs()) > 900
