"""CLI for preprocessing KG datasets.

Builds ID mappings (string → int) and precomputes text embeddings.
Reads from data/{dataset}/raw/, writes to data/{dataset}/processed/.

Usage:
    python -m riemannfm.cli.preprocess data=wikidata_5m embedding=sbert
    python -m riemannfm.cli.preprocess data=fb15k237 embedding=qwen3_embed

Build mini validation dataset from full WikiData5M:
    python -m riemannfm.cli.preprocess data=wikidata_5m preprocess.build_mini=true
"""

import logging

import hydra
from omegaconf import DictConfig

from riemannfm.data.preprocess import build_mini_wikidata_5m, run_preprocess

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.preprocess.build_mini:
        logger.info("Building mini WikiData5M validation dataset...")
        build_mini_wikidata_5m(
            source_dir=cfg.data.data_dir,
            force=cfg.preprocess.force,
        )
        return

    run_preprocess(
        data_dir=cfg.data.data_dir,
        embedding_provider=cfg.embedding.embedding_provider,
        model_name=cfg.embedding.model_name,
        output_dim=cfg.embedding.output_dim,
        batch_size=cfg.preprocess.batch_size,
        max_length=cfg.embedding.get("max_length", 512),
        pooling=cfg.embedding.get("pooling", "mean"),
        api_base=cfg.embedding.get("api_base", None),
        api_key=cfg.embedding.get("api_key", None),
        force=cfg.preprocess.force,
    )
    logger.info("Preprocess complete.")


if __name__ == "__main__":
    main()
