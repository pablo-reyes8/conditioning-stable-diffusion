from pathlib import Path

import yaml


def test_all_yaml_configs_are_valid():
    config_dir = Path("config")
    config_files = sorted(config_dir.rglob("*.yaml"))
    assert config_files, "Expected YAML configuration files under config/."

    for config_file in config_files:
        with config_file.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        assert isinstance(payload, dict), f"Config {config_file} must load as a mapping."
