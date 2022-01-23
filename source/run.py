import  numpy           as  np
import  tensorflow      as  tf

from    preproc.preproc     import  Preprocessor
from    environment         import  Environment
from    agent               import  Agent

from    utils               import  setConfig

class StockTrader(object):
    def __init__(self, EXEC_MODE="train", 
                       CONFIG_FILE_NAME="train_config.json", 
                       RANDOM_SEED=931016
                ):
        # Read configuration file
        config = setConfig(EXEC_MODE, CONFIG_FILE_NAME)

        # Set hardware environment
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
        if config["b_use_gpu"]:
            # Set GPU environment
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ", ".join([str(g) for g in config["usage_gpu_list"]])
            )
        if EXEC_MODE == "test":
            # Fix random seed for testing
            os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
            tf.random.set_seed(RANDOM_SEED)
        else:
            # Set numpy random seed 
            np.random.seed(RANDOM_SEED)

        # Make directories
        [os.mkdir(config["input_folder_path"])
         if not os.path.isdir(config["input_folder_path"]) else None]
        [os.mkdir(config["output_folder_path"])
         if not os.path.isdir(config["output_folder_path"]) else None]

    def train(self):
        # Declare data loader
        print("[1] Load dataset and apply preprocessing")
        preprocessor = Preprocessor(
            TECHNICAL_INDICATOR_LIST=list(),
            USER_DEFINED_FEATURES=dict(),
            bUseTechnicalIndicator=True,
            bUseVolatilityIndex=True,
            bUseTurbulenceIndex=True,
            bUseUserDefinedIndex=True
        )
        
        # Do preprocessing
        dataset = preprocessor.load()
        dataset = preprocessor.apply(dataset)

        # Declare environment
        print("[2] Set environment")
        environment = Environment()

        # Declare agent
        print("[3] Ready for agent")
        agent = Agent()

        # Train agent
        print("[4] Run simulator")
        agent.train()

    def test(self):
        pass
