from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _train_commands(script):
    lines = script.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if '"${PYTHON}" -u train_a2.py \\' in line
    ]
    commands = []
    for start in starts:
        command = []
        for line in lines[start:]:
            command.append(line)
            if len(command) > 1 and not line.rstrip().endswith("\\"):
                break
        commands.append(command)
    return commands


def test_server26_smoke_and_formal_a2_runs_are_fp32():
    script = (ROOT / "scripts" / "run_a2_arm_26.sh").read_text(encoding="utf-8")
    commands = _train_commands(script)

    assert len(commands) == 2
    assert all(any("--no-amp" in line for line in command) for command in commands)


def test_server26_a2_v2_launcher_declares_paired_arms_and_fp32():
    script = (ROOT / "scripts" / "run_a2_v2_arm_26.sh").read_text(
        encoding="utf-8"
    )
    for arm in ("center_pm", "axial_spatial", "axial_delta_k005"):
        assert arm in script
    assert "--adapter-rank 96" in script
    assert "--no-amp" in script
    assert "touch \"${FORMAL_OUT}/completed\"" in script
