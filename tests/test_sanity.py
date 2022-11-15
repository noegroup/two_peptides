

from two_peptides.sanity import EquilibrationStats


def test_create():
    stats = EquilibrationStats(
        density=1.0,
        potential_energy=100.,
        temperature=300.
    )
    print(stats)
