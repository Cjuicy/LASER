from scripts.verify_anchor_propagation import main


def test_anchor_propagation_smoke_script_covers_all_four_modes(capsys):
    main()
    output = capsys.readouterr().out
    for mode in ("depth", "geometry", "layer_atomic", "layer_atomic_split"):
        line = next(
            line for line in output.splitlines()
            if line.startswith(f"[PASS] mode={mode} ")
        )
        assert "g=1.2500" in line
        assert "residual_median=1.0000" in line
        assert "pose_support_ratio=" in line
