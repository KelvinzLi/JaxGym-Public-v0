import pandas as pd

from tqdm import tqdm

def tqdm_callback(iters):
    pbar = tqdm(total = iters, position=0, leave=True)

    def callback(info_dict):
        pbar.update(1)
        pbar.set_postfix(info_dict)

    return callback

class versatile_callback:
    def __init__(self, iters, tqdm_keys = None):
        self.iters = iters
        self.tqdm_keys = tqdm_keys
        
        self.reset()

    def reset(self):
        self._history = []
        self.pbar = tqdm(total = self.iters, position=0, leave=True)

    @property
    def history(self):
        return pd.DataFrame(self._history)

    def __call__(self, info_dict):
        self._history.append(info_dict)

        if self.tqdm_keys is not None:
            for key in self.tqdm_keys:
                if key not in info_dict:
                    print("{} is not inside information provided by trainer".format(key))

            tqdm_info_dict = {k: info_dict[k] for k in self.tqdm_keys}
        else:
            tqdm_info_dict = info_dict

        self.pbar.update(1)
        self.pbar.set_postfix(tqdm_info_dict)

class versatile_callback_v2:
    def __init__(self, iters, tqdm_keys = None, split_train_eval = False):
        self.iters = iters
        self.tqdm_keys = tqdm_keys
        self.split_train_eval = split_train_eval
        
        self.reset()

    def reset(self):
        self._train_history = []
        self._eval_history = []
        self.pbar = tqdm(total = self.iters, position=0, leave=True)

        self.tqdm_info_dict = {}

    @property
    def history(self):
        if not self.split_train_eval:
            return pd.DataFrame(self._train_history)
        else:
            raise ValueError("Please specify if you want train or eval history")

    @property
    def train_history(self):
        return pd.DataFrame(self._train_history)

    @property
    def eval_history(self):
        return pd.DataFrame(self._eval_history)

    def __call__(self, info_dict, from_eval = False):
        if from_eval and self.split_train_eval:
            self._eval_history.append(info_dict)
        else:
            self._train_history.append(info_dict)
            

        available_keys = []

        if self.tqdm_keys is not None:
            for key in self.tqdm_keys:
                if key in info_dict:
                    available_keys.append(key)

            self.tqdm_info_dict = self.tqdm_info_dict | {k: info_dict[k] for k in available_keys}
        else:
            self.tqdm_info_dict = info_dict

        if not from_eval or not self.split_train_eval:
            self.pbar.update(1)
        self.pbar.set_postfix(self.tqdm_info_dict)
        