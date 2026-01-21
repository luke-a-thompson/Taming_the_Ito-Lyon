import argparse

from taming_the_ito_lyon.training.experiment import run_test_from_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the test epoch for a saved run directory"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the saved run directory containing config.toml and best.eqx",
    )
    args = parser.parse_args()
    run_test_from_run_dir(args.run_dir)


if __name__ == "__main__":
    main()
