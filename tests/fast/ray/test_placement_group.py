from types import SimpleNamespace

from miles.ray import placement_group


def _debug_args(*, disaggregated: bool) -> SimpleNamespace:
    return SimpleNamespace(
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        colocate=False,
        critic_num_gpus_per_node=0,
        critic_num_nodes=0,
        debug_rollout_only=True,
        debug_rollout_only_disaggregated=disaggregated,
        debug_train_only=False,
        rollout_num_gpus=8,
        use_critic=False,
    )


def test_debug_rollout_only_uses_only_rollout_bundles_by_default(monkeypatch):
    requested = []

    def fake_create(num_gpus):
        requested.append(num_gpus)
        return "pg", list(range(num_gpus)), list(range(num_gpus))

    monkeypatch.setattr(placement_group, "_create_placement_group", fake_create)

    groups = placement_group.create_placement_groups(_debug_args(disaggregated=False))

    assert requested == [8]
    assert groups["rollout"][1] == list(range(8))


def test_disaggregated_debug_reserves_actor_bundles_before_rollout(monkeypatch):
    requested = []

    def fake_create(num_gpus):
        requested.append(num_gpus)
        return "pg", list(range(num_gpus)), list(range(num_gpus))

    monkeypatch.setattr(placement_group, "_create_placement_group", fake_create)

    groups = placement_group.create_placement_groups(_debug_args(disaggregated=True))

    assert requested == [16]
    assert groups["rollout"][1] == list(range(8, 16))
