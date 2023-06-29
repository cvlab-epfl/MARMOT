import collections
import datetime
import logging
import os
import time
from typing import Dict, List

from pprint import pformat
from configs import config

config.setup_logging(config.get_logging_dict())
log = logging.getLogger('main_logger')  # this is the global logger

def avg_stat_dict(list_of_dict: List[Dict]) -> Dict:
    """Average a list of dictionaries.

    Args:
        list_of_dict (List[Dict]): List of dictionaries.

    Returns:
        Dict: Averaged dictionary.
    """
    results = collections.defaultdict(int)

    for d in list_of_dict:
        for k, v in d.items():
            results[k] += v / len(list_of_dict)

    return results

def flatten_dict(d:Dict, parent_key:str='', sep:str='/') -> Dict:
    """Flatten a nested dictionary.
    
    Args:
        d (dict): Nested dictionary.
        parent_key (str, optional): Parent key. Defaults to ''.
        sep (str, optional): Separator. Defaults to '/'.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def dict_to_string(dict: Dict) -> str:
    """Convert a dictionary to a string.

    Args:
        dict (Dict): Dictionary.

    Returns:
        str: String representation of the dictionary.
    """
    return pformat(dict)


def set_log_level(log_level_name: str) -> None:
    """Set the log level. Log level names are case-insensitive.
    Options are: DEBUG, INFO, WARNING, ERROR.

    Args:
        log_level_name (str): Log level name.
    """
    numeric_level = getattr(logging, log_level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level_name)

    for handler in log.handlers:
        handler.setLevel(numeric_level)