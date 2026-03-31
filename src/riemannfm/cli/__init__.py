"""CLI entry points for RiemannFM.

Loads .env file on import so all CLI commands pick up environment variables
before Hydra resolves ${oc.env:...} interpolations in configs.
"""

from dotenv import load_dotenv

load_dotenv()
