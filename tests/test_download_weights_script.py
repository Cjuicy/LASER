import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download_weights.sh"


def test_download_script_provisions_all_default_pipeline_checkpoints():
    script = Path("scripts/download_weights.sh").read_text(
        encoding="utf-8"
    )
    for checkpoint in (
        "weights/model.safetensors",
        "weights/dino_salad.ckpt",
        "weights/dinov2_vitb14_pretrain.pth",
    ):
        assert checkpoint in script
    assert "mkdir -p weights" in script


@pytest.mark.parametrize(
    "destination",
    (
        "weights/model.safetensors",
        "weights/dino_salad.ckpt",
        "weights/dinov2_vitb14_pretrain.pth",
    ),
)
@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
@pytest.mark.parametrize("interrupted", (False, True))
def test_download_action_never_mutates_external_alias(
    tmp_path, destination, alias_kind, interrupted
):
    target = tmp_path / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / f"external-{target.name}"
    sentinel = b"immutable-external-sentinel"
    external.write_bytes(sentinel)
    if alias_kind == "symlink":
        target.symlink_to(external)
    else:
        os.link(external, target)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "output=\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    --output) output=\"$2\"; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "case \"$output\" in\n"
        "  \"$FAIL_OUTPUT\"|\"$FAIL_OUTPUT\".tmp.*)\n"
        "    printf 'partial-download' > \"$output\"\n"
        "    exit 23\n"
        "    ;;\n"
        "esac\n"
        "printf 'downloaded:%s' \"$output\" > \"$output\"\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAIL_OUTPUT": destination if interrupted else "never-matches",
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0 if interrupted else completed.returncode == 0
    assert external.read_bytes() == sentinel
    assert not tuple(target.parent.glob(f"{target.name}.tmp.*"))
    if interrupted:
        if alias_kind == "symlink":
            assert target.is_symlink()
            assert target.resolve() == external
        else:
            assert os.path.samefile(target, external)
    else:
        assert target.is_file()
        assert not target.is_symlink()
        assert target.read_bytes().startswith(b"downloaded:")
        assert not os.path.samefile(target, external)


@pytest.mark.parametrize(
    "destination",
    (
        "weights/model.safetensors",
        "weights/dino_salad.ckpt",
        "weights/dinov2_vitb14_pretrain.pth",
    ),
)
@pytest.mark.parametrize("destination_kind", ("symlink-directory", "directory"))
def test_download_action_rejects_directory_destination(
    tmp_path, destination, destination_kind
):
    target = tmp_path / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / f"external-dir-{target.name}"
    if destination_kind == "symlink-directory":
        external.mkdir()
        target.symlink_to(external, target_is_directory=True)
    else:
        target.mkdir()
        (target / "sentinel").write_bytes(b"existing-directory-content")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "output=\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in\n"
        "    --output) output=\"$2\"; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        "printf 'downloaded:%s' \"$output\" > \"$output\"\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not tuple(target.parent.glob(f"{target.name}.tmp.*"))
    if destination_kind == "symlink-directory":
        assert target.is_symlink()
        assert target.resolve() == external
        assert not tuple(external.iterdir())
    else:
        assert target.is_dir()
        assert (target / "sentinel").read_bytes() == b"existing-directory-content"
