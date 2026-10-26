import pytest

from msc.training.distributed import RankBatchSampler
from msc.training.sampling import EpochBatchSampler
from msc.training.sampling import InterleavedEpisodeBatchSampler


def test_interleaving_covers_discontiguous_episode_ranges_each_epoch():
    episodes = [[(0, 10), (20, 30)], [(10, 20)], [(30, 53)]]
    batches = list(InterleavedEpisodeBatchSampler(episodes, 7, 0, 16, 42, 2, 3))
    assert sorted(i for batch in batches[:8] for i in batch) == list(range(53))
    assert sorted(i for batch in batches[8:] for i in batch) == list(range(53))
    assert batches[:8] != batches[8:]


def test_interleaving_64_episodes_take_8_then_rotate():
    from collections import Counter
    episodes = [[(i*1000, (i+1)*1000)] for i in range(100)]
    batch = next(iter(InterleavedEpisodeBatchSampler(episodes, 2048, 0, 1, 42)))
    counts = Counter(i//1000 for i in batch)
    assert len(counts) == 64 and set(counts.values()) == {32}
    for first in range(0, 2048, 8):
        assert len({i//1000 for i in batch[first:first+8]}) == 1
    assert len(set(batch)) == 2048


@pytest.mark.parametrize('offset', [0, 1, 4, 7, 8, 9, 15])
def test_interleaving_resume_and_ddp_tail_padding(offset):
    episodes = [[(0, 11)], [(11, 37)], [(37, 53)]]
    args = dict(episodes=episodes, batch_size=8, stop=18, seed=42, active_episodes=2, windows_per_episode=3)
    expected = list(InterleavedEpisodeBatchSampler(start=0, **args))
    assert list(InterleavedEpisodeBatchSampler(start=offset, **args)) == expected[offset:]
    ranks = [list(RankBatchSampler(InterleavedEpisodeBatchSampler(start=0, **args), r, 2, 4)) for r in range(2)]
    for batch, left, right in zip(expected, *ranks):
        assert (left+right)[:len(batch)] == batch
        assert set(left+right) == set(batch)


@pytest.mark.parametrize('size', [*range(1, 34), 63, 64, 65, 255, 256, 257, 1023, 1024, 1025])
def test_global_permutation_covers_irregular_domains(size):
    steps = (size + 10) // 11
    batches = list(EpochBatchSampler(size, 11, 0, steps, seed=42))
    indices = [index for batch in batches for index in batch]
    assert sorted(indices) == list(range(size))


def test_batch_mixes_distant_window_ranges():
    # The old sampler kept the first batch inside one 65,536-window block.
    batch = next(iter(EpochBatchSampler(4 * 65536, 1152, 0, 1, seed=42)))
    assert {index // 65536 for index in batch} == {0, 1, 2, 3}
    assert len(set(batch)) == len(batch)


@pytest.mark.parametrize('offset', [0, 1, 4, 5, 6, 9, 10, 11, 15])
def test_resume_matches_continuous_order_across_epochs(offset):
    args = dict(size=19, batch_size=4, stop=15, seed=42)
    batches = list(EpochBatchSampler(start=0, **args))
    resumed = list(EpochBatchSampler(start=offset, **args))
    assert resumed == batches[offset:]


def test_seed_and_epoch_change_global_order():
    def batches(seed):
        return list(EpochBatchSampler(100, 20, 0, 10, seed))

    first = batches(42)
    assert first == batches(42)
    assert first != batches(43)
    assert first[:5] != first[5:]


def test_two_ranks_reconstruct_global_order_with_only_tail_padding():
    def sampler():
        return EpochBatchSampler(19, 8, 0, 3, seed=42)

    global_batches = list(sampler())
    ranks = [list(RankBatchSampler(sampler(), rank, 2, 4)) for rank in range(2)]
    for i, batch in enumerate(global_batches):
        combined = ranks[0][i] + ranks[1][i]
        assert combined[:len(batch)] == batch
        assert set(combined) == set(batch)
        if len(batch) == 8:
            assert len(set(combined)) == 8
