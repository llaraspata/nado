import os
from argparse import ArgumentParser

import yaml

DEFAULT_CONFIG_FILE = 'config.yaml'


def parse_args_with_config(parser):
    """
    Parse args, taking defaults from the dataset and model entries selected in a yaml config.
    Precedence: command line > model entry > dataset entry > argparse defaults.
    Keys a script doesn't have an argument for are ignored, so one config works for every script.
    """
    config_parser = ArgumentParser(add_help=False) # read these first, before required args are checked
    config_parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE, help='yaml file of datasets/models to test')
    config_parser.add_argument('--dataset', type=str, default=None, help='name of a dataset entry in the config (default: the one selected in the config)')
    config_parser.add_argument('--model', type=str, default=None, help='name of a model entry in the config (default: the one selected in the config)')
    for action in config_parser._actions:
        parser._add_action(action) # so they show up in --help and are accepted by the full parse

    known_args, _ = config_parser.parse_known_args()
    if not os.path.exists(known_args.config):
        if known_args.config != DEFAULT_CONFIG_FILE:
            parser.error('config file {} not found'.format(known_args.config))
        return parser.parse_args()

    with open(known_args.config, 'r') as rf:
        config = yaml.safe_load(rf) or {}

    values = {}
    for section, key in [('datasets', 'dataset'), ('models', 'model')]:
        name = getattr(known_args, key) or config.get(key)
        if name is None:
            continue
        entries = config.get(section) or {}
        if name not in entries:
            parser.error('{} "{}" not in {} (available: {})'.format(key, name, known_args.config, ', '.join(entries)))
        values.update(entries[name] or {})
        values[key] = name

    dests = {action.dest: action for action in parser._actions}
    values = {k: v for k, v in values.items() if k in dests}
    for k in values:
        dests[k].required = False # satisfied by the config, but still overridable from the command line
    parser.set_defaults(**values)

    return parser.parse_args()
