import os
import argparse
import logging
import datetime
import time
from accelerate import Accelerator

import numpy as np
import torch
from LoRA_EA import GLM_Lora

# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Default identifier for unaligned entities
unalign = '999999'


class LogFormatter():
    """
    Custom log formatter to display elapsed time in logs.
    """

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            datetime.timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


class Logger:
    """
    Logger class for handling both file and console logging.

    Attributes:
        log_dir: Directory to save log files
        log_filepath: Full path to log file
        logger: Logger instance
    """

    def __init__(self, log_dir, log_name=None):
        """
        Initialize logger object.

        Args:
            log_dir: Directory to save log files
            log_name: Log file name (optional, default is current time)
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create log formatter
        log_formatter = LogFormatter()

        # Set log file name
        if log_name is None:
            log_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        else:
            log_name += datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.log_filepath = os.path.join(log_dir, log_name)

        # Create console handler and set level to info
        filepath, file_handler = None, None
        if self.log_filepath is not None:
            filepath = '%s' % (self.log_filepath)
            file_handler = logging.FileHandler(self.log_filepath, "a", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)

        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if filepath is not None:
            self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        """Log information message"""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)


class cfg():
    """
    Configuration class for argument parsing and storage.
    """

    def __init__(self):
        self.args = None

    def get_args(self):
        """
        Parse command line arguments.

        Returns:
            Parsed arguments stored in self.args
        """
        parser = argparse.ArgumentParser()

        # Base arguments
        parser.add_argument('--lora', action="store_true", default=False,
                            help="Use LoRA fine-tuning for LLM")
        parser.add_argument('--test', action="store_true", default=False,
                            help="Test model")
        parser.add_argument('--neg_sample', default=1, type=int, choices=[1, 2, 5],
                            help="Negative sampling ratio, 1=Hard negative sampling only")
        parser.add_argument('--dropout', default=0.2, type=float,
                            help="Suggested value: 0.2-0.3")
        parser.add_argument('--epoch', default=1, type=int,
                            help="Number of training epochs")
        parser.add_argument('--dataset', default='FBYG', type=str,
                            choices=["FBYG", "FBDB", "EN_FR_15K_V2", "EN_DE_15K_V2",
                                     "EN_FR_15K", "EN_DE_15K", "ja_en", "zh_en", "fr_en"],
                            help="Dataset name")
        parser.add_argument('--strage', default='matf_l_np', type=str,
                            choices=["l_np", "m_l", "m_l_n", "m_l_np", "m_np", "ma_l_np",
                                     "matf", "matf_l_np", "none", "tri", "nodec", "mr_l_np",
                                     "MM_matf", "MM_matf_l_np"],
                            help="Information filtering strategy")
        parser.add_argument('--rate', default='0.2', type=str,
                            choices=["0.2", "0.3", "0.5", "0.8"],
                            help="Training data ratio")

        self.args = parser.parse_args()


def get_topk(result, s2t, logger):
    """
    Calculate top-k accuracy metrics for entity alignment.

    Args:
        result: Dictionary of {source_entity: [candidate_entities]}
        s2t: Source to target entity alignment mapping
        logger: Logger instance for output
    """
    top_k = [1, 3, 5, 10, 50]
    nofind = 0
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.

    for tar, res in result.items():
        ca = torch.tensor(res)
        try:
            rank = (ca == s2t[int(tar)]).nonzero(as_tuple=False).squeeze().item()
        except:
            nofind += 1
            continue

        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_l2r[i] += 1

    mean_l2r /= len(result)
    mrr_l2r /= len(result)

    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / len(result), 4)

    logger.info(f"Rank result: Hits@{top_k}: {acc_l2r}, MRR: {round(mrr_l2r, 4)}")
    logger.info(f"No alignment found: {nofind}")


def rerank(ill_path, pre_path, result_path, logger):
    """
    Rerank candidate entities based on LLM predictions.

    Args:
        ill_path: Path to alignment pairs file
        pre_path: Path to prediction results
        result_path: Path to original candidate ranking results
        logger: Logger instance for output

    Process:
        1. Load predictions and original results
        2. Rerank candidates by moving predicted aligned entity to first position
        3. Calculate new ranking metrics
    """
    import json
    with open(pre_path, 'r') as f:
        predicts = json.load(f)
    with open(result_path, 'r') as f:
        result = json.load(f)

    def read_file(file_paths):
        """
        Read alignment pairs from file.

        Args:
            file_paths: List of file paths

        Returns:
            List of (source_id, target_id) tuples
        """
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    # Load alignment pairs
    ills = read_file([ill_path + "/ill_ent_ids"])
    s2t = {}
    t2s = {}
    for i, j in ills:
        s2t[i] = j
        t2s[j] = i

    # Remove duplicate candidates
    if len(list(result.values())[0]) != len(set(list(result.values())[0])):
        for tar, res in result.items():
            seen = {}  # Track seen elements
            new_res = []  # Store deduplicated results
            for r in res:
                if r not in seen:
                    seen[r] = 1  # First occurrence
                    new_res.append(r)
                else:
                    seen[r] += 1  # Skip duplicates
            result[tar] = new_res

    logger.info("Test original results")
    get_topk(result, s2t, logger)

    mid = 0
    found = 0
    noin = 0
    npre = 0
    acc = 0
    rerank_res = {}

    # Process predictions for reranking
    for predict in predicts:
        tar, lab, pre, ca, _ = predict.values()
        if pre == lab and lab != unalign:
            acc += 1

        try:
            res = result[str(tar)]
        except:
            noin += 1
            continue

        if pre != unalign and pre in ca:
            try:
                index = res.index(int(pre))
            except:
                npre += 1
                logger.info(f"Don't find {pre} in candidates")
                continue

            if index != 0:
                # Keep original order, move aligned entity to first position
                res.pop(index)
                res.insert(0, int(pre))

            mid += index + 1
            found += 1
            rerank_res[str(tar)] = res
        else:
            rerank_res[str(tar)] = res

    logger.info('Number of targets not in source entities: {0}'.format(noin))
    logger.info('Number of targets not in candidate entities: {0}'.format(npre))

    get_topk(rerank_res, s2t, logger)
    logger.info("Accuracy: {0}".format(round(acc / len(predicts), 4)))
    logger.info("Total test data: {0}".format(len(predicts)))

    try:
        logger.info("Rerank mean index: {0}".format(round(mid / found, 4)))
    except:
        print("Rerank sum of index:", mid, found)
        logger.info("No found")


def train(args):
    """
    Main training and testing pipeline.

    Args:
        args: Parsed command line arguments
    """
    basemodel = "based-Embedding model"
    il = "nil"
    num = 10
    model = 'GLM4'

    root_path = 'model_root'
    file_root = 'data_root'

    # Model selection
    if model == 'GLM4':
        model_path = root_path + "ZhipuAI/glm-4-9b-chat/"
        lora_path = 'GLM4_'
        llm = GLM_Lora(model_path)

    else:
        print('No base model selected')
        exit()

    # Dataset path configuration
    if 'FB' in args.dataset:
        file_dir = file_root + 'wft/code/data/mmkb/{0}'.format(args.dataset + '15K')
    elif 'EN' in args.dataset:
        file_dir = file_root + 'wft/code/data/OpenEA/{0}'.format(args.dataset + '_V2/norm')
    else:
        file_dir = file_root + 'wft/code/data/OpenEA/{0}'.format(args.dataset + '_V2/norm')

    logger = Logger('./logs', basemodel + '_' + '0.2' + '_' + il + "_p1")
    start = 0
    end = num
    data_paths = []

    # Dataset name configuration
    if basemodel != 'MEAformer':
        dataset1 = basemodel + '_' + args.dataset
    else:
        dataset1 = args.dataset

    if 'EN' in args.dataset:
        dataset1 = dataset1 + '_V2_norm'
    elif 'en' in args.dataset:
        dataset1 = basemodel + '_DBP15K_' + args.dataset
    else:
        dataset1 = dataset1 + '15K'

    # Determine number of data slices
    rounds = 2 if args.lora else 5
    if args.strage == 'one' or args.strage == 'none':
        rounds = 1
        end = 20

    # Construct data paths
    for i in range(rounds):
        data_paths.append(
            "result/{0}/{1}/{2}_{3}_{4}/instruct_zero_{5}-{6}".format(basemodel, il, args.dataset, args.rate,
                                                                      args.strage, start, end))
        start = end
        end += num

    print(model, data_paths)
    lora_path = lora_path + basemodel + '_' + args.dataset + '_' + il + "_p1_" + args.strage

    if args.lora:
        logger.info('Training')
        logger.info('LoRA dropout: {0}, Training epochs: {1}, Strategy: {2}'.format(args.dropout, args.epoch,
                                                                                    args.strage))
        print([data_path + "_train.json" for data_path in data_paths[:args.neg_sample]])
        llm.model_lora([data_path + "_train.json" for data_path in data_paths[:args.neg_sample]], lora_path,
                       args, accelerator)

    if args.test:
        logger.info('Evaluation')
        logger.info('Strategy: {0}'.format(args.strage))
        pre_path = "./result/{0}/{1}/{2}_{3}_{4}/eval_{5}_{4}_result.json".format(basemodel, il, args.dataset,
                                                                                  args.rate, args.strage, num,
                                                                                  model)
        # Determine test strategy
        if 'l' in args.strage and 'm' in args.strage:
            test = args.strage.split('_l')[0]
        elif 'l' in args.strage and 'm' not in args.strage:
            test = 'none'
        else:
            test = args.strage

        llm.model_test([data_path.replace(args.strage, test) + "_eval.json" for data_path in data_paths],
                       lora_path + '_' + str(args.epoch), pre_path)

        result_path = "./result/{0}/{1}/{2}_{3}_{1}_left_{4}.json".format(basemodel, il, dataset1, args.rate,
                                                                          'eval')
        rerank(file_dir, pre_path, result_path, logger)


if __name__ == '__main__':
    # Main execution block
    cfg = cfg()
    cfg.get_args()
    # Initialize Accelerator
    accelerator = Accelerator()

    train(cfg.args)
