from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from experiments.window_reference_campaign.cli import main as campaign_main

    return campaign_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
