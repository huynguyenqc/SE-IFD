import csv
import io
import os
import torch
from deep.distributed import apply_gradient_allreduce
from torch import nn, optim
from typing import Any, Callable, Dict, List, Optional, Tuple


def count_parameters(net: nn.Module) -> int:
    cnt = sum([
        params.numel()
        for params in net.parameters()
        if params.requires_grad])
    return cnt


class BaseLogs:
    def __init__(self, init_logs: Optional[Dict[str, float]] = None) -> None:
        self.logs: Dict[str, float] = dict()
        if init_logs is not None:
            self.logs.update(init_logs)

    def get_logs(self) -> Dict[str, float]:
        return self.logs.copy()


class AverageLogs(BaseLogs):
    def __init__(self, init_logs: Optional[Dict[str, float]] = None) -> None:
        super(AverageLogs, self).__init__(init_logs=init_logs)
        self.n: int = 0

    def update(self, logs: Dict[str, float]) -> None:
        for k, v in logs.items():
            self.logs[k] = (self.logs.get(k, 0) * self.n + v) / (self.n + 1)
        self.n += 1


class InstantLogs(BaseLogs):
    def __init__(self, init_logs: Optional[Dict[str, float]] = None) -> None:
        super(InstantLogs, self).__init__(init_logs=init_logs)

    def update(self, logs: Dict[str, float]) -> None:
        self.logs.update(logs)


class Counter:
    def __init__(self, init_n: int = 0) -> None:
        self.n: int = init_n

    def update(self) -> None:
        self.n += 1

    def get_iter_counter(self) -> int:
        return self.n


class CSVLogWriter:
    def __init__(self, file_path: str, field_names: List[str]) -> None:
        self._file_path: str = file_path
        self._field_names: List[str] = field_names.copy()
        self._file_obj: Optional[io.TextIOWrapper] = None
        self._csv_writer: Optional[csv.DictWriter] = None

    def open(self: 'CSVLogWriter'):
        self._file_obj = open(file=self._file_path, mode='w')
        self._csv_writer = csv.DictWriter(
            self._file_obj, fieldnames=self._field_names, delimiter=',', quotechar='"')
        self._csv_writer.writeheader()

    def close(self: 'CSVLogWriter') -> None:
        self._file_obj.close()

    def write_log(self, logs: Dict[str, float]) -> None:
        self._csv_writer.writerow(logs)
        self._file_obj.flush()


def load_model_to_cuda(
        model_class: Callable[..., nn.Module],
        model_configs: Dict[str, Any],
        model_params_filters: Callable[[nn.Module], Any],
        optimiser_class: Callable[..., nn.Module],
        optimiser_configs: Dict[str, Any],
        scheduler_class: Callable[..., optim.lr_scheduler.LRScheduler],
        scheduler_configs: Dict[str, Any],
        model_state_dict: Optional[Dict[str, Any]] = None,
        model_load_state_dict_method_name: str = 'load_state_dict',
        optimiser_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    model = model_class(**model_configs)

    print('Number of parameters of the model: {:,}'.format(count_parameters(model)))

    if model_state_dict is not None:
        assert hasattr(model, model_load_state_dict_method_name)
        model_load_state_dict_method = getattr(model, model_load_state_dict_method_name)
        model_load_state_dict_method(model_state_dict)
        # model.load_state_dict(model_state_dict)
    model = model.cuda()

    if 'num_gpus' in globals():
        global num_gpus
        if num_gpus > 1:
            model = apply_gradient_allreduce(model)

    optimiser = optimiser_class(params=model_params_filters(model), **optimiser_configs)
    if optimiser_state_dict is not None:
        optimiser.load_state_dict(optimiser_state_dict)
        for state in optimiser.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    scheduler = scheduler_class(
        optimizer=optimiser, **scheduler_configs)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    return model, optimiser, scheduler


def load_state_dict_from_path(
        state_dict_path: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], int]:
    if state_dict_path is not None:
        checkpoint_state_dict = torch.load(
            state_dict_path, map_location='cpu')

        return (checkpoint_state_dict.get('model', None),
                checkpoint_state_dict.get('optimiser', None),
                checkpoint_state_dict.get('scheduler', None),
                checkpoint_state_dict.get('epoch', -1))

    return None, None, None, -1


class Checkpointer:
    def __init__(
            self, 
            model: nn.Module, 
            optimiser: optim.Optimizer,
            scheduler: optim.lr_scheduler.LRScheduler,
            save_dir: str,
            epoch_period: int,
            iter_period: Optional[int] = None) -> None:
        self.model_ref: nn.Module = model
        self.optimiser_ref: optim.Optimizer = optimiser
        self.scheduler_ref: optim.lr_scheduler.LRScheduler = scheduler
        self.save_dir: str = save_dir
        self.iter_period: Optional[int] = iter_period
        self.epoch_period: int = epoch_period

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def save_epoch(self, epoch_counter: int) -> None:
        if (epoch_counter + 1) % self.epoch_period == 0:
            torch.save({
                'model': self.model_ref.state_dict(),
                'optimiser': self.optimiser_ref.state_dict(),
                'scheduler': self.scheduler_ref.state_dict(),
                'epoch': epoch_counter},
                os.path.join(self.save_dir, f'epoch_{epoch_counter + 1}.pt'))

    def save_iter(self, iter_counter: int) -> None:
        if self.iter_period is not None:
            if (iter_counter + 1) % self.iter_period == 0:
                torch.save({
                    'model': self.model_ref.state_dict(),
                    'optimiser': self.optimiser_ref.state_dict(),
                    'scheduler': self.scheduler_ref.state_dict()},
                    os.path.join(self.save_dir, f'iter_{iter_counter + 1}.pt'))
