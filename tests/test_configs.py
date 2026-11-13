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
    "model": ["small", "base", "large"],
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
        "fb15k_237",
        "wn18rr",
        "codex_l",
        "yago3_10",
        "wiki27k",
    ],
    "training": ["pretrain"],
    "accelerator": ["auto", "gpu", "cpu", "mps", "ddp"],
    "ablation": [
        "full",
        "no_geok",
        "no_a_r",
        "no_c",
        "no_d_vr",
        "no_d_ve",
        "no_e_v",
        "no_e_r",
    ],
    "logger": ["default", "wandb", "csv", "none"],
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

    def test_wandb(self) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["logger=wandb"])
            assert "wandb" in cfg.logger
            assert "csv" not in cfg.logger

    def test_csv(self) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["logger=csv"])
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
                "num_entities",
            ]:
                assert field in cfg.data, f"data/{variant} missing {field}"


class TestExperimentConfigs:
    """Test that experiment configs compose successfully."""

    _EXP_DIR = Path(CONFIGS_DIR) / "experiment"
    _EXPERIMENTS = sorted(f.stem for f in _EXP_DIR.glob("*.yaml")) if _EXP_DIR.exists() else []

    @pytest.mark.parametrize("exp", _EXPERIMENTS)
    def test_experiment_composes(self, exp: str) -> None:
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[f"experiment={exp}"],
            )
            assert cfg is not None, f"Experiment {exp} failed to compose"


class TestLoggerFilesExist:
    """Ensure all logger config files exist."""

    @pytest.mark.parametrize("name", ["default", "wandb", "csv", "none"])
    def test_logger_yaml_exists(self, name: str) -> None:
        assert (Path(CONFIGS_DIR) / "logger" / f"{name}.yaml").exists()
