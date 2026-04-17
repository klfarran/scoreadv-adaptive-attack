def load_diff_model(ckpt_path, device):
    import numpy as np

    from configs.ve import cifar10_ncsnpp_deep_continuous as configs
    from models import utils as mutils
    from models.ema import ExponentialMovingAverage
    from losses import get_optimizer
    from utils import restore_checkpoint

    # config
    config = configs.get_config()
    config.device = device

    # build model
    diff_model = mutils.create_model(config)

    optimizer = get_optimizer(config, diff_model.parameters())
    ema = ExponentialMovingAverage(diff_model.parameters(),
                                   decay=config.model.ema_rate)

    state = dict(step=0, optimizer=optimizer,
                 model=diff_model, ema=ema)

    # load weights
    state = restore_checkpoint(ckpt_path, state, device)
    ema.copy_to(diff_model.parameters())

    diff_model = diff_model.to(device)
    diff_model.eval()

    # noise levels 
    noise_levels = np.geomspace(
        config.model.sigma_min,
        config.model.sigma_max,
        num=7
    )

    return diff_model, noise_levels