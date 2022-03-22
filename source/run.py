import  os

import  numpy           as  np
import  tensorflow      as  tf

from    preproc.preproc     import  Preprocessor
from    environment         import  Environment, Framework
from    strategy.a2c        import  A2CStrategy

from    utils               import  set_config


class StockTrader(object):
    def __init__(self, mode="train", 
                       config_file_name="train_config.json", 
                       random_seed=931016
                ):
        # Read configuration file
        self.config = set_config(mode, config_file_name)

        # Set hardware environment
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
        if self.config["b_use_gpu"]:
            # Set GPU environment
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ", ".join([str(g) for g in self.config["usage_gpu_list"]])
            )
        if mode == "test":
            # Fix random seed for testing
            os.environ["PYTHONHASHSEED"] = str(random_seed)
            tf.random.set_seed(random_seed)
        else:
            # Set numpy random seed 
            np.random.seed(random_seed)

        # Make directories
        [os.mkdir(self.config["input_folder_path"])
         if not os.path.isdir(self.config["input_folder_path"]) else None]
        [os.mkdir(self.config["output_folder_path"])
         if not os.path.isdir(self.config["output_folder_path"]) else None]

    def train(self):
        # Declare data processor
        print("[1] Declare data processor")
        preprocessor = Preprocessor(
            technical_indicator_list=list(),
            user_defined_features=dict(),
            b_use_technical_indicator=self.config["b_use_technical_indicator"],
            b_use_volatility_index=self.config["b_use_volatility_index"],
            b_use_turbulence_index=self.config["b_use_turbulence_index"],
            b_use_user_defined_index=self.config["b_use_user_defined_index"]
        )
        
        # Load dataset
        print("[2] Load dataset and apply preprocessing")
        input_file_name = (
            self.config["input_folder_path"] 
            + self.config["dataset_file_name"]
        )
        dataset = preprocessor.load_csv(
            tickers=self.config["ticker_list"],
            file_name=input_file_name,
            b_adjusted=True
        )
        dataset = preprocessor.apply(dataset)
        dataset = preprocessor.batch(
            dataset, start_date=20120102, end_date=20121230
        )

        print(len(dataset.index.unique()))

        # Declare environment
        print("[3] Set environment")
        environment = Environment(
            dataset, 
            current_day=0,
            stock_dim=self.config["max_num_stock_hold"],
            hmax_norm=self.config["max_normalized_share_size"],
            init_account_balance=self.config["initital_account_balance"],
            transaction_fee_percent=self.config["transaction_fee_precent"],
            reward_scale=self.config["reward_scaling"]
        )

        # Declare agent
        print("[4] Ready for training")
        simulator = A2CStrategy(
            env=environment, 
            gamma=self.config["gamma"],
            init_learning_rate=self.config["init_learning_rate"],
            num_episode=self.config["num_episode"] 
        )

        # Train agent
        print("[5] Run!")
        simulator.train()

    def trade(self):
        pass


if __name__ == "__main__":
    simulator = StockTrader(
        mode="train",
        config_file_name="train_config.json"
    )

    simulator.train()
