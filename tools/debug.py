import inspect
import torch
from functools import wraps
import logging

def print_forward_shapes(forward_fn):
    def wrapper(self, *args, **kwargs):
        
        DEBUG = False # Set this to True to enable shape printing
        if not DEBUG:
            return forward_fn(self, *args, **kwargs)
        print(f"\n[{self.__class__.__name__}.forward] called")
        sig = inspect.signature(forward_fn)
        param_names = list(sig.parameters.keys())[1:]
        for i, arg in enumerate(args):
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            if isinstance(arg, torch.Tensor):
                print(f"  {name}: shape={arg.shape}")
            else:
                print(f"  {name}: type={type(arg)}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}")
            else:
                print(f"  {k}: type={type(v)}")
        output = forward_fn(self, *args, **kwargs)
        if isinstance(output, torch.Tensor):
            print(f"  Output: shape={output.shape}")
        elif isinstance(output, (tuple, list)):
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  Output[{idx}]: shape={out.shape}")
                else:
                    print(f"  Output[{idx}]: type={type(out)}")
        else:
            print(f"  Output: type={type(output)}")
        return output
    return wrapper 


def log_learning_rate(scheduler):
    def wrapper(*args, **kwargs):
        DEBUG = False
        if not DEBUG:
            return scheduler.step(*args, **kwargs)
        current_lr = scheduler.get_last_lr()[0]
        logging.info(f"Learning Rate: {current_lr}")
        return scheduler.step(*args, **kwargs)
    return wrapper