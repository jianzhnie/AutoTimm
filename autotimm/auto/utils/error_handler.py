"""Internal deep learning framework error handler."""
import contextlib


class TorchErrorCatcher(contextlib.AbstractContextManager):
    """Internal TorchErrorCatcher."""

    def __init__(self):
        self.exc_value = None
        self.hint = ''

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            if 'KeyboardInterrupt' in str(exc_value):
                self.exc_value = 'KeyboardInterrupt'
                return True
            torch_err = str(exc_value)
            if 'GPU is not enabled' in torch_err:
                self.hint += (
                    'Pytorch is not built with GPU support, make sure you have nvidia gpus enabled with '
                    'corresponding torch-cuxxx version installed, where `xxx` is the cuda version. '
                    'You may refer to "https://auto.gluon.ai/stable/install.html" for '
                    'detailed installation guide.\n')
            if 'out of memory' in torch_err:
                self.hint += (
                    'CUDA out of memory error detected. Try reduce `batch_size` in `hyperparameters`, '
                    "e.g., fit(train_data, val_data, hyperparameters={'batch_size': 2}). If the error persists, "
                    'please check if you have other processes occupying too much GPU memory by looking at `nvidia-smi`.\n'
                )
            self.exc_value = torch_err
        else:
            self.exc_value = exc_value
        return True
