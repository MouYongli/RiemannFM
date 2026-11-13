"""CLI for downloading raw KG datasets.

Downloads graph structure and entity text descriptions into data/{dataset}/raw/.
Text embedding precomputation is handled by the preprocess CLI.

Single dataset:
    python -m riemannfm.cli.download data=fb15k_237
    python -m riemannfm.cli.download data=wn18rr

All datasets (Hydra multirun):
    python -m riemannfm.cli.download --multirun \\
        data=wikidata_5m,fb15k_237,wn18rr,codex_l,wiki27k,yago3_10
"""

import logging

import hydra
from omegaconf import DictConfig

from riemannfm.data.pipeline.download import run_pipeline

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(
        slug=cfg.data.slug,
        data_dir=cfg.data.data_dir,
        text_source=cfg.data.text_source,
        force=cfg.download.force,
    )
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
