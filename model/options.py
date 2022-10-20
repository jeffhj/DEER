import argparse
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        
        
    def initialize_parser(self):
        # config file
        self.parser.add_argument('--config', type=str, default='none', help='the config file')
        
        # basic parameters
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--use_checkpoint', type=bool, default=False, help='use checkpoint in the encoder')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")


    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=None,
                        help='total number of step for training, if None then total_steps = batch num * epoch num')
        self.parser.add_argument('--epochs', type=int, default=1,
                        help='total epochs for training, if total_steps is not None, this value will be ignored')
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', type=bool, default=False)


    def add_eval_options(self):
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--eval_print_freq', type=int, default=100,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')
    
    def add_train_options(self):
        # training parameters
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--eval_freq', type=int, default=500,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=5000,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--early_stop_count', type=int, default=10,
                        help='the number of steps for early step')


    def add_model_specific_options(self):
        raise NotImplementedError()
    
    
    def parse(self):
        opt = self.parser.parse_args()
        config_file = opt.config
        with open(config_file, "r") as setting:
            config = yaml.safe_load(setting)
            self.parser.set_defaults(**config)
        opt = self.parser.parse_args()
        return opt


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        model_dir:Path = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

class FiDOptions(Options):
    def add_eval_options(self):
        super().add_eval_options()
        self.parser.add_argument('--write_results', type=bool, default=False, help='save results')
        self.parser.add_argument('--write_crossattention_scores', type=bool, default=False, 
                        help='save dataset with cross-attention scores')


    def initialize_parser(self):
        super().initialize_parser()
        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        
        
    def add_model_specific_options(self):
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--n_context', type=int, default=1)
        self.parser.add_argument('--text_maxlength', type=int, default=200, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_sent', type=bool, default=False, help='no sentence in context')
        self.parser.add_argument('--no_path', type=bool, default=False, help='no path information')
        self.parser.add_argument('--duplicate_sample', type=bool, default=True, help='use duplicated samples to align context passages')
        
        