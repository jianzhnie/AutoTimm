def update_cfg(cfg, udict):
    cfg.unfreeze()
    cfg.update(udict)
    cfg.freeze()
