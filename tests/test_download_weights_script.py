from pathlib import Path


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
