import os
import yaml
import torch
import itertools
import pandas as pd
from copy import deepcopy
from torchinfo import summary
from datasets.ml_1m import ML1MDataset
from datasets.ml_100k import ML100KDataset
from pytorch_lightning import seed_everything
from datasets.amazon import AmazonIndustrialDataset, AmazonVideoGamesDataset, AmazonCDDataset
from model.muffin import Muffin

PROJECT_NAME = 'recsys'
EXPERIMENT_ROOT = 'experiments'
RAW_DATASET_ROOT_FOLDER = 'data'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    ML100KDataset.code(): ML100KDataset,
    AmazonCDDataset.code(): AmazonCDDataset,
    AmazonIndustrialDataset.code(): AmazonIndustrialDataset,
    AmazonVideoGamesDataset.code(): AmazonVideoGamesDataset,
}


def set_all_seeds(seed=42):
    # Python
    import random
    random.seed(seed)

    # NumPy
    import numpy as np
    np.random.seed(seed)

    # PyTorch
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # CUDA deterministic operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dataloader_factory(args): # dataloader_factory(args={'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0,...}) getting argument
    """Get dataloader based on dataset code and model code"""
    dataset = DATASETS[args['dataset_code']](args)
    from dataloader.dataloader import SequentialDataloader
    dataloader = SequentialDataloader(args, dataset)

    return dataloader.get_pytorch_dataloaders()


def train(args, export_root=None): #first check in __main__ we get train({'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0,...}, experiments/random_testing/ml-1m/bsarec/c_bsarec_l_0.001_d_64_i_0_a_gelu_h_2_l_2_c_3_a_0.9_d_0.4_d_0.4)
    seed_everything(args['seed']) # seed_everything(42)
    set_all_seeds(args['seed'])  # set_all_seeds(42)

    if export_root is None:
        if not os.path.exists("experiments/main"):
            os.makedirs("experiments/main")

        export_root = EXPERIMENT_ROOT + '/main/'+ args['model_code'] + '/' + args['dataset_code'] + \
                      '_' + str(args['weight_decay']) + '_' + str(args['dropout']) + '_' + str(args['attn_dropout'])

    trn, val, tst = dataloader_factory(args)  # dataloader_factory(args={'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0,...}) passing argument
    if args['model_code'] == 'grit':
        pass
        # from model.gritrec import GRITRec
        # model = GRITRec(args)
        # summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    # elif args['model_code'] == 'grit_a':
        # from model.gritrec_a import GRITRecAblation
        # model = GRITRecAblation(args)
        # summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'cirm':
        pass
        # from model.cirm import CIRM
        # model = CIRM(args)
        # summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'glintru':
        from model.glintru import GLINTRU
        model = GLINTRU(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'bsarec':
        from model.bsarec import BSARec
        model = BSARec(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'muffin':
        from model.muffin import Muffin
        model = Muffin(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'morexpert3':
        from model.morbeta import MoRTransformerModel
        from model.morexpert3 import MixtureOfExpertsWrapper
        experts = [MoRTransformerModel(args) for _ in range(args['num_experts'])]
        model = MixtureOfExpertsWrapper(args, experts)

        # model.transform_to_mor_expert(config)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'morexpert1':
        from model.mormoeV1 import MoRTransformerModel
        from model.morexpert1 import MixtureOfExpertsWrapper
        experts = [MoRTransformerModel(args) for _ in range(args['num_experts'])]
        model = MixtureOfExpertsWrapper(args, experts)

        # model.transform_to_mor_expert(config)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'morexpert5':
        from model.mormoeV1 import MoRTransformerModel
        from model.morexpert5 import MixtureOfExpertsWrapper
        experts = [MoRTransformerModel(args) for _ in range(args['num_experts'])]
        model = MixtureOfExpertsWrapper(args, experts)

        # model.transform_to_mor_expert(config)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'diffurec':
        from model.diffurec import DiffuRec
        model = DiffuRec(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'lrurec':
        from model.lrurec import LRURec
        model = LRURec(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'linrec':
        from model.linrec import LinRec
        model = LinRec(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'fmlprec':
        from model.fmlprec import FMLPRec
        model = FMLPRec(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'duorec':
        from model.duorec import DuoRec
        model = DuoRec(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'tisasrec':
        from model.TiSASRec import TiSASRecModel
        print(args)
        model = TiSASRecModel(args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    elif args['model_code'] == 'unirec':
        print(f"Implementation of BaseLine {args['model_code']} is Under Process! :(  ")
        exit()

    elif args['model_code'] == 'ticoserec':
        from model.TiCoSeRec import SASRec
        model = SASRec(args)
        #model = TiCoSeRec(model_sasrec,trn, val, tst, args)
        summary(model, dtypes=[torch.long], verbose=2, col_width=16)

    else:
        raise ValueError(f"Unknown model code: {args['model_code']}")

    if args['model_code'] in ['grit', 'grit_a', 'cirm']:
        from trainer.trainers import SequentialTrainer
        trainer = SequentialTrainer(args, model, trn, val, tst, export_root)
    else:
        from trainer.trainers import BaselineTrainer
        trainer = BaselineTrainer(args, model, trn, val, tst, export_root)

    trainer.train()
    valid_metrics  = trainer.get_best_val_metrics()
    test_metrics = trainer.test()
    return valid_metrics, test_metrics


if __name__ == "__main__":
    # List available config files
    base_yaml = "config/base.yaml"

    with open(base_yaml, 'r') as f:
        base_config = yaml.safe_load(f)   # {'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0, 'min_rating': 0, 'max_seq_length': 50, 'split': 'leave_one_out', 'sliding_window_size': 1, 'val_batch_size': 256, 'test_batch_size': 256, 'train_batch_size': 256, 'num_epochs': 10, 'optimizer': 'AdamW', 'max_grad_norm': 5.0, 'weight_decay': 0.01, 'adam_epsilon': 1e-09, 'early_stopping': True, 'best_metric': 'Recall@10', 'early_stopping_patience': 10, 'metric_ks': [1, 5, 10, 20, 50], 'time_scales': [1, 60, 3600, 86400, 604800]}


    print('******************** Dataset Selection ********************')
    dataset_code = {'1': 'ml-1m', '2': 'ml-100k', '3': 'video_games', '4': 'cds_and_vinyl', '5': 'industrial_and_scientific'}
    print("Available datasets:", dataset_code)
    data_code = input("Enter the dataset code: ").strip()
    base_config['dataset_code'] = dataset_code[data_code]    #{'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0, 'min_rating': 0, 'max_seq_length': 50, 'split': 'leave_one_out', 'sliding_window_size': 1, 'val_batch_size': 256, 'test_batch_size': 256, 'train_batch_size': 256, 'num_epochs': 10, 'optimizer': 'AdamW', 'max_grad_norm': 5.0, 'weight_decay': 0.01, 'adam_epsilon': 1e-09, 'early_stopping': True, 'best_metric': 'Recall@10', 'early_stopping_patience': 10, 'metric_ks': [1, 5, 10, 20, 50], 'time_scales': [1, 60, 3600, 86400, 604800], 'dataset_code': 'ml-1m'}

    model_yaml_dir = "config/optimal"
  #  print(os.listdir(model_yaml_dir))  #['bsarec.yaml', 'duorec.yaml', 'fmlprec.yaml', 'glintru.yaml', 'linrec.yaml', 'lrurec.yaml']

    print('******************** Model Selection ********************')
    available_configs = [f.replace('.yaml', '') for f in os.listdir(model_yaml_dir) if f.endswith('.yaml')]  #Available datasets: ['bsarec', 'duorec', 'fmlprec', 'glintru', 'linrec', 'lrurec']
    print("Available models:", ", ".join(available_configs)) #Available models: bsarec, duorec, fmlprec, glintru, linrec, lrurec
    model_code = input("Enter the model code: ").strip()
    model_yaml = os.path.join(model_yaml_dir, f"{model_code}.yaml") #config/optimal\bsarec.yaml

    with open(model_yaml, 'r') as f:
        param_grid = yaml.safe_load(f)      #{'model_code': ['bsarec'], 'lr': [0.001], 'embed_dim': [64], 'num_prev_items': [0], 'act': ['gelu'], 'num_heads': [2], 'num_layers': [2], 'c': [3], 'alpha': [0.9], 'dropout': [0.4], 'attn_dropout': [0.4]}

    if "-" in base_config['dataset_code']:
        dataset_initials = base_config['dataset_code']  # keep as is like ml-1m
    else:
        dataset_initials = "_".join([part[0] for part in base_config['dataset_code'].split("_")])

    if not os.path.exists(f"experiments/random_testing/{dataset_initials}"):
        os.makedirs(f"experiments/random_testing/{dataset_initials}")

    RESULT_FILE = f'{EXPERIMENT_ROOT}/random_testing/{dataset_initials}/{param_grid["model_code"][0]}_results.xlsx'

    # Load existing results if they exist
    if os.path.exists(RESULT_FILE):
        existing_val_df = pd.read_excel(RESULT_FILE, sheet_name='val')     #  model_code     lr  embed_dim  ...  Recall@1     MRR@1    NDCG@1
                                                                              # 0     bsarec  0.001         64  ...  0.052974  0.052974  0.052974
        existing_test_df = pd.read_excel(RESULT_FILE, sheet_name='test')   #  model_code     lr  embed_dim  ...  Recall@1     MRR@1    NDCG@1
                                                                              # 0     bsarec  0.001         64  ...  0.048845  0.048845  0.048845
    else:
        existing_val_df = pd.DataFrame()
        existing_test_df = pd.DataFrame()

    # Add missing keys to existing DataFrames and initialize with 0
    param_keys = list(param_grid.keys()) #['model_code', 'lr', 'embed_dim', 'num_prev_items', 'act', 'num_heads', 'num_layers', 'c', 'alpha', 'dropout', 'attn_dropout']
    if not existing_val_df.empty:
        for key in param_keys:
            if key not in existing_val_df.columns:
                existing_val_df[key] = 0
                existing_test_df[key] = 0

    # Keep a set of existing combinations to avoid duplicates
    existing_keys = set()
    if not existing_val_df.empty:
        key_cols = list(param_grid.keys())  #['model_code', 'lr', 'embed_dim', 'num_prev_items', 'act', 'num_heads', 'num_layers', 'c', 'alpha', 'dropout', 'attn_dropout']
        existing_keys = set(tuple(row[k] for k in key_cols) for _, row in existing_val_df.iterrows()) #{('bsarec', 0.001, 64, 0, 'gelu', 2, 2, 3, 0.9, 0.4, 0.4)}

    # Prepare to collect new results
    val_records = []
    test_records = []

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]   #[{'model_code': 'bsarec', 'lr': 0.001, 'embed_dim': 64, 'num_prev_items': 0, 'act': 'gelu', 'num_heads': 2, 'num_layers': 2, 'c': 3, 'alpha': 0.9, 'dropout': 0.4, 'attn_dropout': 0.4}]

    for combo in all_combinations:
        combo_key = tuple(combo[k] for k in keys)
        print(f"Running combo: {combo}")  #Running combo: {'model_code': 'bsarec', 'lr': 0.001, 'embed_dim': 64, 'num_prev_items': 0, 'act': 'gelu', 'num_heads': 2, 'num_layers': 2, 'c': 3, 'alpha': 0.9, 'dropout': 0.4, 'attn_dropout': 0.4}
        config = deepcopy(base_config)
        config.update(combo) #{'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0, 'min_rating': 0, 'max_seq_length': 50, 'split': 'leave_one_out', 'sliding_window_size': 1, 'val_batch_size': 256, 'test_batch_size': 256, 'train_batch_size': 256, 'num_epochs': 10, 'optimizer': 'AdamW', 'max_grad_norm': 5.0, 'weight_decay': 0.01, 'adam_epsilon': 1e-09, 'early_stopping': True, 'best_metric': 'Recall@10', 'early_stopping_patience': 10, 'metric_ks': [1, 5, 10, 20, 50], 'time_scales': [1, 60, 3600, 86400, 604800], 'dataset_code': 'ml-1m', 'model_code': 'bsarec', 'lr': 0.001, 'embed_dim': 64, 'num_prev_items': 0, 'act': 'gelu', 'num_heads': 2, 'num_layers': 2, 'c': 3, 'alpha': 0.9, 'dropout': 0.4, 'attn_dropout': 0.4}
        parts = [f"{k.split('_')[-1][0]}_{v}" for k, v in combo.items()]
        export_path = (os.path.join(f"experiments/random_testing/{config['dataset_code']}/{config['model_code']}/", "_".join(parts)))  # experiments/random_testing/ml-1m/bsarec/c_bsarec_l_0.001_d_64_i_0_a_gelu_h_2_l_2_c_3_a_0.9_d_0.4_d_0.4

        val_metrics, tst_metrics = train(config, export_path) #train({'seed': 42, 'device': 'cuda', 'num_workers': 0, 'min_uc': 5, 'min_sc': 5, 'overlap': 0,...}, experiments/random_testing/ml-1m/bsarec/c_bsarec_l_0.001_d_64_i_0_a_gelu_h_2_l_2_c_3_a_0.9_d_0.4_d_0.4)

        # Add hyperparams to metrics
        val_entry = {**combo, **val_metrics}
        test_entry = {**combo, **tst_metrics}

        val_records.append(val_entry)
        test_records.append(test_entry)

        # Append to existing DataFrames
        existing_val_df = pd.concat([existing_val_df, pd.DataFrame([val_entry])], ignore_index=True)
        existing_test_df = pd.concat([existing_test_df, pd.DataFrame([test_entry])], ignore_index=True)

        # Save after each run
        with pd.ExcelWriter(RESULT_FILE, engine='openpyxl', mode='w') as writer:
            existing_val_df.to_excel(writer, sheet_name='val', index=False)
            existing_test_df.to_excel(writer, sheet_name='test', index=False)
