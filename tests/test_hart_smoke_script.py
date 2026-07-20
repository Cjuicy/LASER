from scripts.verify_anchor_propagation import main


def test_anchor_propagation_smoke_script_covers_all_four_modes(capsys):
    main()
    output = capsys.readouterr().out
    for mode in ("depth", "geometry", "layer_atomic", "layer_atomic_split"):
        assert f"[PASS] mode={mode}" in output
