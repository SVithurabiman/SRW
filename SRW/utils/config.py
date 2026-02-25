from types import SimpleNamespace
import yaml
import os

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def to_namespace(d):
    return SimpleNamespace(
        **{k: to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def load_config(data_cfg=None,  exp_cfg=None, config_type ="Train"):
    cfg = {}
    cfg.update(load_yaml(exp_cfg))
    cfg["data"] = load_yaml(data_cfg)
    if config_type == "Test":
        cfg["model"] = load_yaml(os.path.join(cfg["path"],"config.yaml"))
    return to_namespace(cfg)
