import yaml

from configs.paths import PROJECT_ROOT
from configs.random_seed_config import RandomSeedConfig

YAML_PATH = PROJECT_ROOT / "configs" / "random_seed.yaml"

with open(YAML_PATH, "r") as f:
    _raw_random_seed_config = yaml.safe_load(f)


def load_random_seed_config():
    return RandomSeedConfig(**_raw_random_seed_config["global_seed"])
