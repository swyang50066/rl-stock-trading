import  os

import  numpy           as  np
import  tensorflow      as  tf

from    preproc.preproc     import  Preprocessor
from    environment         import  Environment, Framework
from    agent               import  Agent

from    utils               import  setConfig


class StockTrader(object):
    def __init__(self, EXEC_MODE="train", 
                       CONFIG_FILE_NAME="train_config.json", 
                       RANDOM_SEED=931016
                ):
        # Read configuration file
        self.config = setConfig(EXEC_MODE, CONFIG_FILE_NAME)

        # Set hardware environment
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
        if self.config["b_use_gpu"]:
            # Set GPU environment
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ", ".join([str(g) for g in self.config["usage_gpu_list"]])
            )
        if EXEC_MODE == "test":
            # Fix random seed for testing
            os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
            tf.random.set_seed(RANDOM_SEED)
        else:
            # Set numpy random seed 
            np.random.seed(RANDOM_SEED)

        # Make directories
        [os.mkdir(self.config["input_folder_path"])
         if not os.path.isdir(self.config["input_folder_path"]) else None]
        [os.mkdir(self.config["output_folder_path"])
         if not os.path.isdir(self.config["output_folder_path"]) else None]

    def train(self):
        # Declare data loader
        print("[1] Set data loader")
        preprocessor = Preprocessor(
            TECHNICAL_INDICATOR_LIST=list(),
            USER_DEFINED_FEATURES=dict(),
            bUseTechnicalIndicator=self.config["b_use_technical_indicator"],
            bUseVolatilityIndex=self.config["b_use_volatility_index"],
            bUseTurbulenceIndex=self.config["b_use_turbulence_index"],
            bUseUserDefinedIndex=self.config["b_use_user_defined_index"]
        )
        
        # Do preprocessing
        print("[2] Load dataset and apply preprocessing")
        dataset = preprocessor.load(
            DATASET_FINE_NAME=self.config["dataset_file_name"]
        )
        dataset = preprocessor.apply(dataset)

        # Declare environment
        print("[3] Set environment")
        environment = Environment(
            dataset, 
            day=0
            STOCK_DIM=self.config["max_num_stock_hold"]
            HMAX_NORMALIZE=self.config["max_normalized_share_size"]
            INITIAL_ACCOUNT_BALACE=self.config["initital_account_balance"]
            TRANSACTION_FEE_PERCENT=self.config["transaction_fee_precent"]
            REWARD_SCALING=self.config["reward_scaling"]
        )

        # Declare agent
        print("[4] Ready for training")
        agent = Agent()

        # Train agent
        print("[5] Run!")
        agent.train()

    def trade(self):
        pass
