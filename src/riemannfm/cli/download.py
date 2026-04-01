"""CLI for downloading raw KG datasets.

Downloads graph structure and entity text descriptions into data/{dataset}/raw/.
Text embedding precomputation is handled by the preprocess CLI.

Single dataset:
    python -m riemannfm.cli.download data=fb15k237
    python -m riemannfm.cli.download data=wn18rr

All datasets:
    python -m riemannfm.cli.download download.all=true
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from riemannfm.data.pipeline.download import ALL_DATASETS, run_pipeline

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    force = cfg.download.force

    if cfg.download.all:
        logger.info("Downloading ALL datasets...")
        for slug in ALL_DATASETS:
            data_cfg = _load_data_config(slug)
            if data_cfg is None:
                logger.warning(f"No data config found for slug={slug}, skipping.")
                continue
            run_pipeline(
                slug=slug,
                data_dir=data_cfg["data_dir"],
                text_source=data_cfg["text_source"],
                force=force,
            )
    else:
        slug = cfg.data.slug
        run_pipeline(
            slug=slug,
            data_dir=cfg.data.data_dir,
            text_source=cfg.data.text_source,
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
            data_cfg = OmegaConf.load(yaml_file)
            if data_cfg.get("slug") == slug:
                return {
                    "data_dir": data_cfg.get("data_dir", f"data/{slug}"),
                    "text_source": data_cfg.get("text_source", ""),
                }
        except Exception:
            continue

    return None


if __name__ == "__main__":
    main()
