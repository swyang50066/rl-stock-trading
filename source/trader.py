import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from preproc.preproc import Preprocessor
from environment import Environment, Framework

from strategy.a2c import A2CStrategy

from utils import set_config, make_directory, join_path_item


class StockTraderWrapper(ABC):
    """Superclass for thetrader sucbclasses"""
    def __init__(self, mode, config_file_name, random_seed=931016):
        # Parse configuration file
        self.mode = mode
        self.config = set_config(mode, config_file_name)
        
        # Make directories and set data file
        make_directory(self.config["input_folder_path"])
        make_directory(self.config["output_folder_path"])

        # Set hardware environment
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        if self.config["b_use_gpu"]:    # Set GPU environment
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(
                [str(g) for g in self.config["usage_gpu_list"]]
            )
        if mode == "test":    # Fix random seed for reproductivity
            os.environ["PYTHONHASHSEED"] = str(random_seed)
            tf.random.set_seed(random_seed)
    
    def load_dataset(self):
        """Load dataset"""
        # Declare preprocessor
        preprocessor = Preprocessor(
            technical_indicator_list=list(),
            user_defined_features=dict(),
            b_use_technical_indicator=self.config[
                "b_use_technical_indicator"
            ],
            b_use_volatility_index=self.config["b_use_volatility_index"],
            b_use_turbulence_index=self.config["b_use_turbulence_index"],
            b_use_user_defined_index=self.config["b_use_user_defined_index"],
        )

        # Load dataset
        if self.config["b_use_saved_dataset"]:
            dataset = preprocessor.load_csv(
                tickers=self.config["ticker_list"],
                file_name=join_path_item(
                    self.config["input_folder_path"],
                    self.config["dataset_file_name"]
                ),
                b_adjusted=True,
            )
        else:
            dataset = preprocessor.load_yahoo_finance(
                tickers=self.config["ticker_list"],
                start_date=self.config["start_date"],
                end_date=self.config["end_date"]
            )

        # Apply preprocessing
        dataset = preprocessor.apply(dataset)

        # Get batch dataset
        dataset = preprocessor.batch(
            dataset,
            start_date=self.config["start_date"],
            end_date=self.config["end_date"]
        )

        return dataset

    def load_env(self, dataset):
        """"Load environment"""
        env = Environment(
            dataset,
            current_day=0,
            stock_dim=self.config["max_num_stock_hold"],
            hmax_norm=self.config["max_normalized_share_size"],
            init_account_balance=self.config["initital_account_balance"],
            transaction_fee_percent=self.config["transaction_fee_precent"],
            reward_scale=self.config["reward_scaling"],
        )

        if self.mode == "train":
            return env
        else:
            return Framework(
                env=env, 
                b_initial=True,
                turbulence_threshold=self.config["turbulence_threshold"]
            )

    @abstractmethod
    def setup(self):
        """Set up trader"""
        pass

    @abstractmethod
    def run(self):
        """Run algorithm"""
        pass


class A2CStockTrader(StockTraderWrapper):
    """A2C algorithm based trader"""
    def __init__(self, mode, config_file_name):
        super(A2CStockTrader, self).__init__(mode, config_file_name)

    def setup(self):
        """Set up trader"""
        # Load dataset
        dataset = self.load_dataset()

        # Load environment
        env = self.load_env(dataset)

        # Declare trader
        self.trader = A2CStrategy(
            env=env,
            gamma=self.config["gamma"],
            init_learning_rate=self.config["init_learning_rate"],
            num_episode=self.config["num_episode"],
        )

    def run(self):
        """Run algorithm"""
        if self.mode:
            self.trader.train()
        else:
            self.trader.trade()
