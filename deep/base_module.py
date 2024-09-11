from torch import nn
from pydantic import BaseModel


def args_kwargs_to_kwargs(key_list, args, kwargs):
    return {
        **{_arg: _arg_val for _arg, _arg_val in zip(key_list, args)},
        **kwargs}


class ModuleInterface:
    """ 
    An interface for PyTorch module with some features:
        1) Serialisable class for configuration loading / saving
        2) Configurations for Low-Rank Adaptation

    Example:
    ```
    class MyModule(nn.Module, ModuleInterface):
        class ConstructorArgs(ModuleInterface.ConstructorArgs):
            dim: int

        def __init__(self, **kwargs) -> None:
            # Module initialisation first, then interface initialisation
            ModuleInterface.__init__(self, **kwargs)
            nn.Module.__init__(self)
    ```
    """
    ConstructorArgs = BaseModel

    def __init__(self, *args, **kwargs) -> None:
        assert isinstance(self, nn.Module), \
            'Only apply module interface to `nn.Module`!'

        # Store serialisatble config as class object
        self._configs = self.ConstructorArgs(
            **args_kwargs_to_kwargs(
                key_list=list(self.ConstructorArgs.model_fields.keys()),
                args=args,
                kwargs=kwargs))

    @property
    def get_config_object(self) -> BaseModel:
        return self._configs.model_copy()


def sanity_test():
    class MyModule(nn.Module, ModuleInterface):
        class ConstructorArgs(ModuleInterface.ConstructorArgs):
            dim: int

        def __init__(self, **kwargs) -> None:
            ModuleInterface.__init__(self, **kwargs)
            nn.Module.__init__(self)

            self.net = nn.Linear(self._configs.dim, self._configs.dim)

    my_module = MyModule(dim=10)
    print(my_module)


if __name__ == '__main__':
    sanity_test()
