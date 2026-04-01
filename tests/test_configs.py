"""Tests for Hydra config composition and _target_ consistency.

Validates that all YAML configs compose correctly with the root config,
that _target_ fields are present where expected, and that interpolations
resolve without errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize_config_dir

if TYPE_CHECKING:
    from omegaconf import DictConfig

CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Config groups and their variants.
CONFIG_GROUPS: dict[str, list[str]] = {
    "model": ["rieformer_small", "rieformer_base", "rieformer_large"],
    "manifold": [
        "product_h_s_e",
        "h_only",
        "s_only",
        "e_only",
        "h_e",
        "s_e",
        "fixed_curvature",
    ],
    "flow": ["joint", "continuous_only", "discrete_only"],
    "data": [
        "wikidata_5m",
        "wikidata_5m_mini",
        "fb15k237",
        "wn18rr",
        "codex_l",
        "yago3_10",
        "wiki27k",
    ],
    "training": ["pretrain", "finetune"],
    "accelerator": ["auto", "gpu", "cpu", "mps", "ddp"],
    "ablation": [
        "full",
        "no_mrope",
        "no_geok",
        "no_mrope_geok",
        "no_ath",
        "no_edge_self",
        "no_cross",
        "no_text_cond",
    ],
    "logger": ["default", "wandb_only", "csv_only", "none"],
}

# Groups that must have _target_.
GROUPS_WITH_TARGET = ["model", "manifold", "flow", "data", "training", "accelerator"]

# Expected _target_ values per group.
EXPECTED_TARGETS: dict[str, str] = {
    "model": "riemannfm.models.riemannfm.RiemannFM",
    "manifold": "riemannfm.manifolds.product.RiemannFMProductManifold",
    "flow": "riemannfm.flow.joint_flow.RiemannFMJointFlow",
    "data": "riemannfm.data.datamodule.RiemannFMDataModule",
    "training": "riemannfm.models.lightning_module.RiemannFMPretrainModule",
    "accelerator": "lightning.pytorch.Trainer",
}


@pytest.fixture(scope="module")
def default_cfg() -> DictConfig:
    """Compose the default root config."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(config_name="config")


class TestDefaultComposition:
    """Test that the default config composes successfully."""

    def test_compose_succeeds(self, default_cfg: DictConfig) -> None:
        assert default_cfg is not None

    def test_has_all_groups(self, default_cfg: DictConfig) -> None:
        for group in [
            "paths",
            "model",
            "manifold",
            "flow",
            "data",
            "training",
            "ablation",
            "accelerator",
            "eval",
            "download",
            "preprocess",
            "logger",
        ]:
            assert group in default_cfg, f"Missing config group: {group}"

    def test_seed_and_project(self, default_cfg: DictConfig) -> None:
        assert default_cfg.seed == 42
        assert default_cfg.project_name == "riemannfm"

    def test_paths_log_dir_defined(self, default_cfg: DictConfig) -> None:
        """Regression: log_dir must be defined for logger configs."""
        assert "log_dir" in default_cfg.paths


class TestTargetFields:
    """Test that _target_ is present and correct in config groups."""

    @pytest.mark.parametrize("group", GROUPS_WITH_TARGET)
    def test_target_present(self, default_cfg: DictConfig, group: str) -> None:
        cfg_group = default_cfg[group]
        assert "_target_" in cfg_group, f"{group}/ missing _target_"

    @pytest.mark.parametrize("group", GROUPS_WITH_TARGET)
    def test_target_value(self, default_cfg: DictConfig, group: str) -> None:
        actual = default_cfg[group]["_target_"]
        expected = EXPECTED_TARGETS[group]
        assert actual == expected, f"{group}._target_ = {actual}, expected {expected}"


class TestConfigVariants:
    """Test that each variant in each group composes successfully."""

    @pytest.mark.parametrize(
        "group,variant",
        [
            (group, variant)
            for group, variants in CONFIG_GROUPS.items()
            for variant in variants
        ],
    )
    def test_variant_composes(self, group: str, variant: str) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[f"{group}={variant}"])
            assert cfg is not None
            assert group in cfg


class TestLoggerGroup:
    """Test the standardized logger config group."""

    def test_default_has_wandb_and_csv(self, default_cfg: DictConfig) -> None:
        assert "wandb" in default_cfg.logger
        assert "csv" in default_cfg.logger

    def test_default_wandb_has_target(self, default_cfg: DictConfig) -> None:
        assert "_target_" in default_cfg.logger.wandb

    def test_default_csv_has_target(self, default_cfg: DictConfig) -> None:
        assert "_target_" in default_cfg.logger.csv

    def test_none_is_empty(self) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["logger=none"])
            assert len(cfg.logger) == 0

    def test_wandb_only(self) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["logger=wandb_only"])
            assert "wandb" in cfg.logger
            assert "csv" not in cfg.logger

    def test_csv_only(self) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["logger=csv_only"])
            assert "csv" in cfg.logger
            assert "wandb" not in cfg.logger


class TestDeadParametersRemoved:
    """Regression: dead parameters must not exist."""

    def test_no_use_mrope_in_model(self, default_cfg: DictConfig) -> None:
        assert "use_mrope" not in default_cfg.model

    def test_no_use_mrope_in_ablation(self, default_cfg: DictConfig) -> None:
        assert "use_mrope" not in default_cfg.ablation

    def test_no_activation_checkpointing(self, default_cfg: DictConfig) -> None:
        assert "activation_checkpointing" not in default_cfg.training

    def test_no_zinc_data(self) -> None:
        zinc_path = Path(CONFIGS_DIR) / "data" / "zinc.yaml"
        assert not zinc_path.exists()

    def test_no_loggers_string(self, default_cfg: DictConfig) -> None:
        """Regression: old 'loggers' string hack must not exist."""
        assert "loggers" not in default_cfg


class TestFlowParamsComplete:
    """Regression: flow configs must have all parameters."""

    @pytest.mark.parametrize("variant", CONFIG_GROUPS["flow"])
    def test_flow_has_radius_and_sigma(self, variant: str) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[f"flow={variant}"])
            assert "radius_h" in cfg.flow
            assert "sigma_e" in cfg.flow
            assert "disable_continuous" in cfg.flow
            assert "disable_discrete" in cfg.flow
            assert "t_max" in cfg.flow


class TestDataParamsComplete:
    """Regression: all data configs must have required fields."""

    @pytest.mark.parametrize("variant", CONFIG_GROUPS["data"])
    def test_data_has_required_fields(self, variant: str) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[f"data={variant}"])
            for field in [
                "slug",
                "num_edge_types",
                "max_nodes",
                "max_hops",
                "num_workers",
                "dim_text_emb",
                "num_entities",
            ]:
                assert field in cfg.data, f"data/{variant} missing {field}"


class TestExperimentConfigs:
    """Test that experiment configs compose successfully."""

    @pytest.fixture(scope="class")
    def experiment_files(self) -> list[str]:
        exp_dir = Path(CONFIGS_DIR) / "experiment"
        if not exp_dir.exists():
            return []
        return [f.stem for f in exp_dir.glob("*.yaml")]

    def test_experiments_compose(self, experiment_files: list[str]) -> None:
        for exp in experiment_files:
            with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"+experiment={exp}"],
                )
                assert cfg is not None, f"Experiment {exp} failed to compose"


class TestNoOldLoggerConfig:
    """Ensure old logger files are removed."""

    def test_no_wandb_yaml(self) -> None:
        assert not (Path(CONFIGS_DIR) / "logger" / "wandb.yaml").exists()

    def test_no_csv_yaml(self) -> None:
        assert not (Path(CONFIGS_DIR) / "logger" / "csv.yaml").exists()

    def test_default_yaml_exists(self) -> None:
        assert (Path(CONFIGS_DIR) / "logger" / "default.yaml").exists()

    def test_none_yaml_exists(self) -> None:
        assert (Path(CONFIGS_DIR) / "logger" / "none.yaml").exists()
