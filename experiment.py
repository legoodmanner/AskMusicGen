from omegaconf import DictConfig
# from .abstract_experiment import AbstractExperiment
import torch
from tqdm.auto import tqdm

from models.lightning import ProbeModule
from models.gen_models import get_gen_model
from data import get_dataModule

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler, SimpleProfiler


from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import os
from omegaconf import OmegaConf


PROGRESS_BAR = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3f",
    )
)


class Experiment:
    def __init__(self, config: DictConfig):

        # Init the rest of configs according to the abstract class
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_seed(config.experiment.seed)
        self.setup_config()
        self.dl = self.setup_dataloader()
        self.setup_model()


    def setup_seed(self, seed: int):

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def setup_config(self):
        task = self.config.experiment.task # str or None
        gen_model = self.config.experiment.gen_model
        if task:
            config_task = os.path.join('configs', 'tasks', self.config.experiment.task) + '.yaml'
            if os.path.isfile(config_task):
                print(f'Task {task} build up data configurations from {config_task}')
                self.config = OmegaConf.merge(self.config, OmegaConf.load(config_task))
            else:
                raise NameError(f'Cannot find {config_task}.yaml')
        else:
            raise ValueError('Cannot build the task correctly, make sure there is either config.experiment.task ')
        if gen_model:
            config_gen = os.path.join('configs', 'gens', gen_model) + '.yaml'
            if os.path.isfile(config_gen):
                print(f'Task {task} build up data configurations from {config_gen}')
                self.config = OmegaConf.merge(self.config, OmegaConf.load(config_gen))
            else:
                raise NameError(f'Cannot find {config_gen}.yaml')
        else:
            raise ValueError('Cannot build the task correctly, make sure there is either config.experiment.gen')
            
            
    def setup_model(self):

        # Initiate audio and text encoder
        # TODO: Support more audio and text encoder for future experiments
        gen_model = get_gen_model(self.config)
        self.model = ProbeModule(gen_model, config=self.config)
        # Get the parameters for the model
        return None
    
    def setup_logger(self):

        # No need for tensorboard logger
        exp_config = self.config.experiment
        if exp_config is None:
            from lightning.pytorch.loggers import TensorBoardLogger
            return TensorBoardLogger(os.path.join(exp_config.logger.output_dir, exp_config.logger.project), name=self.config.experiment.name)
        else:
            if exp_config.logger.type == "wandb":
                # Wandblogger
                from lightning.pytorch.loggers import WandbLogger
                wandb_logger = WandbLogger(project=exp_config.logger.project, save_dir= exp_config.logger.output_dir, name=exp_config.logger.name)
                return wandb_logger
            else:
                raise NotImplementedError("Logger type not supported")


    def setup_dataloader(self, sampling=False):
        data_module = get_dataModule(self.config)
        return data_module
    
    def setup_early_stopping(self):
        
        early_stop_callback = EarlyStopping(
            monitor="val_loss", 
            min_delta=0.00, 
            patience=30, 
            verbose=False, 
            mode="min"
        )

        return early_stop_callback
    

    def dry_run(self):

        trainer = L.Trainer(
            max_epochs = 1,
            accelerator = "gpu",
            limit_train_batches= 10,
            limit_val_batches= 2,
            log_every_n_steps = 10,
            default_root_dir="outputs/dry_run",
            profiler = SimpleProfiler(filename="profiler-report"),
            callbacks=[PROGRESS_BAR],
            precision = "16-mixed",
            # detect_anomaly=True
        )

        trainer.fit(model = self.model, 
            train_dataloaders = self.dl.train_dataloader(),
            val_dataloaders = self.dl.val_dataloader())

        print(f"\033[32m Dry Run Finished \033[0m")

    def overfit(self):

        trainer = L.Trainer(
            accelerator = "gpu",
            overfit_batches=5,
            default_root_dir="outputs/overfit",
            log_every_n_steps= self.config.training.log_step,
            callbacks = [PROGRESS_BAR],
            precision = "16-mixed",
            gradient_clip_val=0.5
        )

        trainer.fit(model = self.model, 
            train_dataloaders = self.dl.train_dataloader(),
            )

        print(f"\033[32m Overfit Finished \033[0m")
    

    def train(self):
        
        early_stop_callback = self.setup_early_stopping()

        logger = self.setup_logger()

        # if self.config.environment.gpu_num > 1:
        #         print(f"\033[93m Using DDP; GPU Num: {self.config.environment.gpu_num} \033[0m")
        #         trainer = L.Trainer(
        #             max_epochs=self.config.training.epochs, 
        #             accelerator="gpu",
        #             default_root_dir=self.config.experiment.output_dir,
        #             callbacks=[PROGRESS_BAR],
        #             # logger = logger,
        #             log_every_n_steps= self.config.training.log_step,
        #             val_check_interval= self.config.training.val_check_step,
        #             precision = "16-mixed",
        #             limit_val_batches= 20,
        #             devices=self.config.environment.gpu_num, 
        #             strategy="ddp",
        #             gradient_clip_val=0.5,
        #             gradient_clip_algorithm='value'
        #         )
            
        # else:

        trainer = L.Trainer(
            max_epochs=self.config.training.epochs, 
            accelerator="gpu",
            default_root_dir=self.config.experiment.logger.output_dir,
            callbacks=[PROGRESS_BAR],
            logger = logger,
            log_every_n_steps= self.config.training.log_step,
            # val_check_interval= self.config.training.val_check_interval,
            check_val_every_n_epoch= 5,
            precision = "16-mixed",
            limit_val_batches= 20,
            gradient_clip_val=0.5,
            gradient_clip_algorithm='value',
            accumulate_grad_batches= self.config.training.accumulate_grad_batches,
        )

        ckpt_path = self.config.training.get('start_from_ckpt')
        if ckpt_path is not None:
            print("Start from: " + str(ckpt_path))
        else:
            print("Start from scratch")


        trainer.fit(model = self.model, 
            train_dataloaders = self.dl.train_dataloader(),
            val_dataloaders = self.dl.val_dataloader(),
            ckpt_path =ckpt_path)

        print('Training done')

    

    # def resume(self, ckpt_path: str):

    #     trainer = L.Trainer()
    #     trainer.fit(self.model, ckpt_path=ckpt_path)

    # def load_from_checkpoint(self, checkpoint_path: str):
    #     if not checkpoint_path.endswith('ckpt'):
    #         checkpoint_path = os.path.join(checkpoint_path, 'checkpoints', os.listdir(os.path.join(checkpoint_path, 'checkpoints'))[0])
    #     self.model = LitEncodecTransformer.load_from_checkpoint(checkpoint_path, model=self.model.model)
        
        return None 
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'c', type=str, default=None, help='path of the configuration file'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.c)
    exp = Experiment(base_config)
    if args.dry_run:
        exp.dry_run()
    else:
        exp.train()


       
