"""CLI for downloading and preprocessing KG datasets.

Downloads graph structure, extracts entity text descriptions, and
precomputes text embeddings using Hydra configuration.

Single dataset:
    python -m riemannfm.cli.download data=fb15k237 text_encoder=sbert
    python -m riemannfm.cli.download data=wn18rr text_encoder=xlm_roberta

All datasets:
    python -m riemannfm.cli.download download.all=true text_encoder=sbert
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from riemannfm.data.download import ALL_DATASETS, run_pipeline

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Download config:\n{OmegaConf.to_yaml(cfg.download)}")

    text_encoder_name = cfg.text_encoder.model_name
    dim_text_emb = cfg.text_encoder.output_dim
    batch_size = cfg.download.batch_size
    device = cfg.download.device
    force = cfg.download.force

    if cfg.download.all:
        # Process all datasets: iterate over registry, resolve each data config
        logger.info("Downloading ALL datasets...")
        for slug in ALL_DATASETS:
            # Load data config for this slug to get data_dir and text_source
            data_cfg = _load_data_config(slug)
            if data_cfg is None:
                logger.warning(f"No data config found for slug={slug}, skipping.")
                continue
            run_pipeline(
                slug=slug,
                data_dir=data_cfg["data_dir"],
                text_source=data_cfg["text_source"],
                text_encoder=text_encoder_name,
                dim_text_emb=dim_text_emb,
                batch_size=batch_size,
                device=device,
                force=force,
            )
    else:
        # Single dataset from Hydra config
        slug = cfg.data.slug
        run_pipeline(
            slug=slug,
            data_dir=cfg.data.data_dir,
            text_source=cfg.data.text_source,
            text_encoder=text_encoder_name,
            dim_text_emb=dim_text_emb,
            batch_size=batch_size,
            device=device,
            force=force,
        )

    logger.info("Download complete.")


def _load_data_config(slug: str) -> dict | None:
    """Load a data config YAML by dataset slug.

    Scans configs/data/*.yaml to find the one matching the given slug.
    Returns a dict with data_dir and text_source, or None if not found.
    """
    from pathlib import Path

    config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "data"
    if not config_dir.exists():
        return None

    for yaml_file in config_dir.glob("*.yaml"):
        try:
            cfg = OmegaConf.load(yaml_file)
            if cfg.get("slug") == slug:
                return {
                    "data_dir": cfg.get("data_dir", f"data/{slug}"),
                    "text_source": cfg.get("text_source", ""),
                }
        except Exception:
            continue

    return None


if __name__ == "__main__":
    main()
