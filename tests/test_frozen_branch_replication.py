from pathlib import Path
import pytest
from scripts.server.run_frozen_branch_seed_replication import pretraining_commands


def test_replication_keeps_fixed_heads_budgets_and_matched_random_arms():
    commands = pretraining_commands(Path("out"), Path("audit"), Path("nt.npz"), 314159, [0, 1, 2, 3])
    assert len(commands) == 4
    for name, command in commands.items():
        assert command[command.index("--seed") + 1] == "314159"
        assert command[command.index("--arms") + 1] == ("bidirectional_linear" if name.startswith("sequence") else "bidirectional_default")
        assert ("--random-encoder-control" in command) == name.endswith("random")
        assert ("--node-feature-cache" in command) == name.startswith("sequence")
    for seed, gpus in [(42, [0, 1, 2, 3]), (314159, [0, 1, 1, 2])]:
        with pytest.raises(ValueError):
            pretraining_commands(Path("out"), Path("audit"), Path("nt.npz"), seed, gpus)
