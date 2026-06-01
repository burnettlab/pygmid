import datetime
import logging
import logging.config
import os
from pathlib import Path

import coloredlogs
import yaml
from auto_all import public
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging

logger = logging.getLogger(__name__)


def timedelta_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return datetime.timedelta(**fields)


yaml.SafeLoader.add_constructor("!timedelta", timedelta_constructor)


@public
def setup_logging(
    default_path=Path(__file__).parent / "logging.yaml",
    default_level=logging.INFO,
    env_key="LOG_CFG",
):
    if value := os.getenv(env_key, None):
        path = value
    else:
        path = default_path

    path = Path(path)

    if path.exists():
        with open(path, "rt") as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                logger.debug(
                    f"Error in Logging Configuration: {e}. Using default configs"
                )
                logging.basicConfig(level=default_level)
    else:
        logger.debug("Failed to load configuration file. Using default configs")
        logging.basicConfig(level=default_level)
