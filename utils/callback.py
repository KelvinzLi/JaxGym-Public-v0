from tqdm import tqdm

def tqdm_callback(iters):
    pbar = tqdm(total = iters, position=0, leave=True)

    def callback(info_dict):
        pbar.update(1)
        pbar.set_postfix(info_dict)

    return callback