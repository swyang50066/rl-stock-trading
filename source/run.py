import os
import sys
import argparse

from trader import (
    A2CStockTrader,
)


def parse_argument(args):
    """parse user arguments"""
    parser = argparse.ArgumentParser(description="Execution Information")

    # Add arguments
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="execution mode",
    )
    parser.add_argument(
        "--config_file_name",
        dest="config_file_name",
        type=str,
        default="train_config.json",
        help="configuration file name",
    )
    parser.add_argument(
        "--algo",
        dest="algo",
        type=str,
        default="a2c",
        choices=["a2c"],
        help="RL algorithm type",
    )
    parser.set_defaults(render=False)

    return parser.parse_args(args)


def runner(args=None):
    """main"""
    # Set trader
    print("[1] Set trader algorithm")
    info = parse_argument(args)
    if info.algo == "a2c":
        trader = A2CStockTrader(
            mode=info.mode, config_file_name=info.config_file_name
        )
    else:
        raise NotImplementedError("RL algorithm type is not given!")

    print("[2] Load dataset and environment")
    trader.setup()

    print("[3] Run simulation")
    trader.run()


if __name__ == "__main__":
    runner()
