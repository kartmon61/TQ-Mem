"""TQ-Mem: query-adaptive temporal kernel retrieval for long-term LLM agent memory."""
__version__ = "0.1.0"

from .env import load_env as _load_env  # .env -> os.environ once per process (credentials never on the command line)
_load_env()
