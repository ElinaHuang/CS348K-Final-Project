from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="session")
def final_config():
    from generate_prompts import load_config
    return load_config("configs/grammar_config.yaml")


@pytest.fixture(scope="session")
def final_prompt_data(final_config):
    from generate_prompts import generate_all
    return generate_all(final_config)
