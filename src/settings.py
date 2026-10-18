"""Настройки kedro-проекта"""

import os

from kedro.config import OmegaConfigLoader

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # Параметры разнесены по файлам: source.yml -> params:data, training.yml -> params:training
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**", "source*", "training*"],
    },
    # ${runenv:NAME, default} — значение из переменной окружения или default
    "custom_resolvers": {
        "runenv": lambda name, default=None: os.environ.get(name, default),
    },
}
