import contextlib
import datetime
import numpy as np
import os
import random
import shutil
import torch
import yaml
from pydantic import BaseModel, Field
from torch import nn, optim
from torch.utils import data as torch_data
from typing import Any, Dict, Generator, List, Optional, Union, Literal

from deep.distributed import init_distributed
from deep.utils import (
    AverageLogs, InstantLogs, Checkpointer, Counter, CSVLogWriter,
    load_model_to_cuda, load_state_dict_from_path)
from utils.snapshot_src import snapshot

from ..data_utils import get_dataset, get_dataset_names
from ..models import get_model, get_model_names
from .media_logging import MediaDataLogger


# Global variable for distributed training
num_gpus = None
rank = None
group_name = None


DATASET_NAMES = tuple(get_dataset_names())
DATASET_CLASSES = tuple(get_dataset(ds_name) for ds_name in DATASET_NAMES)
DATASET_CONSTRUCTORS = tuple(ds_cls.ConstructorArgs for ds_cls in DATASET_CLASSES)

MODEL_NAMES = tuple(get_model_names())
MODEL_CLASSES = tuple(get_model(m_name) for m_name in MODEL_NAMES)
MODEL_CONSTRUCTORS = tuple(m_cls.ConstructorArgs for m_cls in MODEL_CLASSES)


class DatasetConfigs(BaseModel):
    name: Literal[DATASET_NAMES]                                          # type: ignore
    configs: Union[DATASET_CONSTRUCTORS] = Field(discriminator='name')    # type: ignore


def init_dataset(ds_cfgs: DatasetConfigs) -> Union[DATASET_CLASSES]:      # type: ignore
    return get_dataset(ds_cfgs.name)(**ds_cfgs.configs.model_dump())


class ModelConfigs(BaseModel):
    name: Literal[MODEL_NAMES]                                            # type: ignore
    configs: Union[MODEL_CONSTRUCTORS] = Field(discriminator='name')      # type: ignore


class DataLoaderConfigs(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    shuffle: Optional[bool] = None
    pin_memory: bool = False
    drop_last: bool = False


class TrainConfigs(BaseModel):
    n_epochs: int
    iter_per_checkpoint: Optional[int] = None
    epoch_per_val: int
    epoch_per_checkpoint: int
    optimiser: Dict[str, Any]
    scheduler: Dict[str, Any]
    train_dataset: DatasetConfigs
    train_dataloader: DataLoaderConfigs
    validation_dataset: DatasetConfigs
    validation_dataloader: DataLoaderConfigs
    auto_mix_precision: bool = False


class EpochValidator(MediaDataLogger):
    def __init__(
            self,
            dataloader: torch_data.DataLoader,
            model: nn.Module,
            save_dir: str,
            log_writer: CSVLogWriter,
            epoch_period: int,
            fs: int = 16000) -> None:
        super(EpochValidator, self).__init__()

        self.model_ref: nn.Module = model
        self.val_dataloader: torch_data.DataLoader = dataloader
        self.save_dir: str = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.csv_log_writer: int = log_writer
        self.epoch_period: int = epoch_period
        self.fs: int = fs

    def validate(self, epoch: Optional[int] = None) -> None:
        if epoch is None:   # Force validation
            target_dir = os.path.join(self.save_dir, 'epoch_final')
            figure_target_dir = os.path.join(target_dir, 'figures') 
            waveform_target_dir = os.path.join(target_dir, 'waveform') 
            os.mkdir(target_dir)
            os.mkdir(figure_target_dir)
            os.mkdir(waveform_target_dir)

            logs = AverageLogs()

            for i, batch_data in enumerate(self.val_dataloader):
                y_bt = batch_data[0].cuda()
                x_bt = batch_data[1].cuda()

                self.model_ref.eval()
                val_data = self.model_ref.validate(x_bt=x_bt, y_bt=y_bt, epoch=epoch)

                # Write numerical values to CSV
                if val_data.get('numerical') is not None:
                    logs.update(val_data['numerical'])

                # Plot spectrum and mask (images)
                if val_data.get('spectrum') is not None:
                    self.plot_dict_spectrum(val_data['spectrum'], figure_target_dir, i)
                if val_data.get('mask') is not None:
                    self.plot_dict_mask(val_data['mask'], figure_target_dir, i)

                # Write wav files
                if val_data.get('waveform') is not None:
                    self.write_dict_wav(val_data['waveform'], waveform_target_dir, i, fs=self.fs)
        elif (epoch + 1) % self.epoch_period == 0:
            # Remove log directory of previous validation
            if os.path.exists(os.path.join(self.save_dir, f'epoch_{epoch + 1 - self.epoch_period}')):
                shutil.rmtree(os.path.join(self.save_dir, f'epoch_{epoch + 1 - self.epoch_period}'), ignore_errors=True)

            target_dir = os.path.join(self.save_dir, f'epoch_{epoch + 1}')
            assert not os.path.exists(target_dir)
            figure_target_dir = os.path.join(target_dir, 'figures') 
            waveform_target_dir = os.path.join(target_dir, 'waveform') 
            os.mkdir(target_dir)
            os.mkdir(figure_target_dir)
            os.mkdir(waveform_target_dir)

            logs = AverageLogs()

            for i, batch_data in enumerate(self.val_dataloader):
                y_bt = batch_data[0].cuda()
                x_bt = batch_data[1].cuda()

                self.model_ref.eval()
                val_data = self.model_ref.validate(x_bt=x_bt, y_bt=y_bt, epoch=epoch)

                # Write numerical values to CSV
                if val_data.get('numerical') is not None:
                    logs.update(val_data['numerical'])

                # Plot spectrum and mask (images)
                if val_data.get('spectrum') is not None:
                    self.plot_dict_spectrum(val_data['spectrum'], figure_target_dir, i)
                if val_data.get('mask') is not None:
                    self.plot_dict_mask(val_data['mask'], figure_target_dir, i)

                # Write wav files
                if val_data.get('waveform') is not None:
                    self.write_dict_wav(val_data['waveform'], waveform_target_dir, i, fs=self.fs)

            self.csv_log_writer.write_log(logs.get_logs())


class TrainContext:
    def __init__(
            self,
            log_dir: str,
            configs: Dict[str, Any],
            field_names: List[str],
            model: nn.Module,
            optimiser: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler.LRScheduler],
            train_dataloader: torch_data.DataLoader,
            val_dataloader: torch_data.DataLoader,
            iter_save_period: int,
            epoch_val_period: int,
            epoch_save_period: int) -> None:
        
        self.log_dir: str = log_dir
        self.configs: Dict[str, Any] = configs
        self.field_names: List[str] = field_names
        self.model: nn.Module = model
        self.optimiser: optim.Optimizer = optimiser
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = scheduler
        self.train_dataloader: torch_data.DataLoader = train_dataloader
        self.val_dataloader: torch_data.DataLoader = val_dataloader
        self.iter_save_period: int = iter_save_period
        self.epoch_val_period: int = epoch_val_period
        self.epoch_save_period: int = epoch_save_period

        self.epoch_train_log_writer: Optional[CSVLogWriter] = None
        self.iter_train_log_writer: Optional[CSVLogWriter] = None
        self.epoch_val_log_writer: Optional[CSVLogWriter] = None

        self.checkpointer: Optional[Checkpointer] = None
        self.validator: Optional[EpochValidator] = None

        self.iter_counter: Optional[Counter] = None

    def start_context(self):
        global rank
        if rank == 0:

            # Create directory
            assert not os.path.exists(self.log_dir), 'The folder has already existed!'
            os.makedirs(self.log_dir)

            with open(os.path.join(self.log_dir, 'configs.yml'), 'w') as f_configs:
                yaml.safe_dump(data=self.configs, stream=f_configs, default_flow_style=False)

            # Snapshot source code before training
            snapshot_src_dir = os.path.join(self.log_dir, 'src')
            os.mkdir(snapshot_src_dir)
            snapshot(
                snapshot_src_dir, 
                source_packages = [
                    'deep', 'recipes/se_ifd', 'utils', 'sample_data',
                    'run.sh', 'requirements.txt'])

            # Create logging context
            self.epoch_train_log_writer = CSVLogWriter(
                os.path.join(self.log_dir, 'epoch_train_log.csv'), self.field_names)
            self.epoch_train_log_writer.open()

            self.iter_train_log_writer = CSVLogWriter(
                os.path.join(self.log_dir, 'iter_train_log.csv'), self.field_names)
            self.iter_train_log_writer.open()

            self.epoch_val_log_writer = CSVLogWriter(
                os.path.join(self.log_dir, 'epoch_val_log.csv'), self.field_names)
            self.epoch_val_log_writer.open()

            # Checkpointer
            self.checkpointer = Checkpointer(
                model=self.model, optimiser=self.optimiser, scheduler=self.scheduler,
                save_dir=os.path.join(self.log_dir, 'checkpoints'),
                iter_period=self.iter_save_period, epoch_period=self.epoch_save_period)

            # Validator
            self.validator = EpochValidator(
                dataloader=self.val_dataloader, model=self.model, 
                save_dir=os.path.join(self.log_dir, 'val_logs'), 
                log_writer=self.epoch_val_log_writer,
                epoch_period=self.epoch_val_period,
                fs=self.configs['train']['train_dataset']['configs']['sr'])

        # Iter counter
        self.iter_counter = Counter()

    def end_context(self):
        global rank
        if rank == 0:
            self.epoch_train_log_writer.close()
            self.iter_train_log_writer.close()
            self.epoch_val_log_writer.close()

        # Reset 
        self.epoch_train_log_writer = None
        self.iter_train_log_writer = None
        self.epoch_val_log_writer = None

        self.checkpointer = None
        self.validator = None
        self.iter_counter = None


@contextlib.contextmanager
def train_wrapper(
        log_dir: str,
        configs: Dict[str, Any],
        field_names: List[str],
        model: nn.Module,
        optimiser: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_dataloader: torch_data.DataLoader,
        val_dataloader: torch_data.DataLoader,
        iter_save_period: int,
        epoch_val_period: int,
        epoch_save_period: int
) -> Generator[TrainContext, None, None]:

    train_context = TrainContext(
        log_dir=log_dir, configs=configs, field_names=field_names,
        model=model, optimiser=optimiser, scheduler=scheduler,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        iter_save_period=iter_save_period, epoch_val_period=epoch_val_period,
        epoch_save_period=epoch_save_period)

    try:
        train_context.start_context()
        yield train_context
    finally:
        train_context.end_context()


@contextlib.contextmanager
def epoch_wrapper(epoch: int, train_context: TrainContext) -> Generator[AverageLogs, None, None]:
    global rank

    epoch_logs = AverageLogs()
    yield epoch_logs

    if rank == 0:
        train_context.checkpointer.save_epoch(epoch_counter=epoch)
        train_context.epoch_train_log_writer.write_log(epoch_logs.get_logs())
        train_context.validator.validate(epoch)


@contextlib.contextmanager
def iter_wrapper(train_context: TrainContext) -> Generator[InstantLogs, None, None]:
    global rank
    instant_logs = InstantLogs()
    yield instant_logs

    if rank == 0:
        train_context.checkpointer.save_iter(train_context.iter_counter.get_iter_counter())
        train_context.iter_counter.update()
        train_context.iter_train_log_writer.write_log(instant_logs.get_logs())

    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_function(
        log_parent_dir: str,
        identifier: str,
        train_configs: TrainConfigs,
        model_configs: ModelConfigs,
        pretrain_model_configs: Optional[ModelConfigs] = None,
        pretrain_model_path: Optional[str] = None,
        init_state_dict_path: Optional[str] = None,
        num_gpus_: int = 1,
        rank_: int = 0,
        group_name_: str = '') -> str:

    global num_gpus
    global rank
    global group_name

    num_gpus = num_gpus_
    rank = rank_
    group_name = group_name_

    if num_gpus > 1:
        init_distributed(rank=rank, num_gpus=num_gpus, group_name=group_name, dist_backend='nccl', dist_url='tcp://localhost:54321')

    train_dataset = init_dataset(train_configs.train_dataset)
    val_dataset = init_dataset(train_configs.validation_dataset)

    train_sampler = torch_data.DistributedSampler(
        train_dataset, shuffle=train_configs.train_dataloader.shuffle
    ) if num_gpus > 1 else None

    train_dataloader_configs = train_configs.train_dataloader.model_copy()
    train_dataloader_configs.batch_size //= num_gpus
    if train_sampler is not None:
        train_dataloader_configs.shuffle = False 

    # torch_g = torch.Generator()
    # torch_g.manual_seed(3407)

    train_dataloader = torch_data.DataLoader(
        dataset=train_dataset, sampler=train_sampler, 
        # worker_init_fn=seed_worker, generator=torch_g,
        **train_dataloader_configs.model_dump())
    val_dataloader = torch_data.DataLoader(
        dataset=val_dataset, **train_configs.validation_dataloader.model_dump())

    if pretrain_model_path is None:
        (model_state_dict,
         optimiser_state_dict,
         scheduler_state_dict,
         previous_epoch) = load_state_dict_from_path(init_state_dict_path)

        model, optimiser, scheduler = load_model_to_cuda(
            model_class=get_model(model_configs.name),
            model_configs=model_configs.configs.model_dump(),
            model_params_filters=lambda net: net.get_training_parameters(),
            optimiser_class=optim.Adam,
            optimiser_configs=train_configs.optimiser,
            scheduler_class=optim.lr_scheduler.OneCycleLR,
            scheduler_configs=train_configs.scheduler,
            model_state_dict=model_state_dict,
            optimiser_state_dict=optimiser_state_dict,
            scheduler_state_dict=scheduler_state_dict)
    else:
        pretrain_model_state_dict, _, _, _ = load_state_dict_from_path(pretrain_model_path)
        optimiser_state_dict, scheduler_state_dict, previous_epoch = None, None, -1

        model, optimiser, scheduler = load_model_to_cuda(
            model_class=get_model(model_configs.name),
            model_configs=model_configs.configs.model_dump(),
            model_params_filters=lambda net: net.get_training_parameters(),
            model_load_state_dict_method_name='load_state_dict_from_pretrain_state_dict',
            optimiser_class=optim.Adam,
            optimiser_configs=train_configs.optimiser,
            scheduler_class=optim.lr_scheduler.OneCycleLR,
            scheduler_configs=train_configs.scheduler,
            model_state_dict={
                'model_cls': get_model(pretrain_model_configs.name),
                'configs': pretrain_model_configs.configs.model_dump(),
                'state_dict': pretrain_model_state_dict},
            optimiser_state_dict=optimiser_state_dict,
            scheduler_state_dict=scheduler_state_dict)

    print('Number of parameters for optimisation: {:,}'.format(sum([
        params.numel() 
        for params in model.get_training_parameters() 
        if params.requires_grad])))

    if train_configs.auto_mix_precision:
        scaler = torch.cuda.amp.GradScaler()

    log_dir = os.path.join(log_parent_dir, '{}_{}'.format(
        identifier, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    list_fields = model.output_logging_keys

    with train_wrapper(
            log_dir=log_dir,
            configs={'model': model_configs.model_dump(), 'train': train_configs.model_dump()},
            field_names=list_fields,
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            iter_save_period=train_configs.iter_per_checkpoint,
            epoch_val_period=train_configs.epoch_per_val,
            epoch_save_period=train_configs.epoch_per_checkpoint
    ) as train_context:
        if previous_epoch + 1 >= train_configs.n_epochs:
            train_context.validator.validate()  # Force validation
        for epoch in range(previous_epoch + 1, train_configs.n_epochs):
            with epoch_wrapper(epoch, train_context) as epoch_logs:
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                for batch_data in train_dataloader:
                    y_bt = batch_data[0]
                    x_bt = batch_data[1]

                    with iter_wrapper(train_context) as iter_logs:
                        model.train()

                        model.zero_grad()
                        optimiser.zero_grad()

                        if train_configs.auto_mix_precision:
                            # Train with GradScaler
                            with torch.cuda.amp.autocast(enabled=True):
                                x_bt = x_bt.cuda()
                                y_bt = y_bt.cuda()
                                loss, logs = model(x_bt=x_bt, y_bt=y_bt, epoch=epoch)

                            scaler.scale(loss).backward()
                            scaler.step(optimiser)
                            scale_value = scaler.get_scale()
                            scaler.update()
                            skip_scheduler = scale_value > scaler.get_scale()
                            if not skip_scheduler and scheduler is not None:
                                scheduler.step()
                        else:
                            # Train without GradScaler
                            x_bt = x_bt.cuda()
                            y_bt = y_bt.cuda()
                            loss, logs = model(x_bt=x_bt, y_bt=y_bt, epoch=epoch)
                            loss.backward()
                            optimiser.step()
                            if scheduler is not None:
                                scheduler.step()

                        iter_logs.update(logs)
                        epoch_logs.update(logs)

    if rank == 0:
        torch.save({
            'model': model.state_dict()},
            os.path.join(log_dir, 'checkpoints', 'epoch_final.pt'))
    return log_dir
