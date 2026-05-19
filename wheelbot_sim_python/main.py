"""
Top-level Hydra entry point for the wheelbot_sim_python package.

Training (default):
    python -m wheelbot_sim_python.main

Video evaluation:
    python -m wheelbot_sim_python.main video_eval=true eval.log_dir=exp/test/27752/2026.05.05/135429
    python -m wheelbot_sim_python.main video_eval=true eval.log_dir=... eval.iteration=3 eval.track_seed=42
"""

import jax
import omegaconf
import hydra
from wheelbot_sim_python.training_scripts.brax_infoprop_train import main as infoprop_main
from wheelbot_sim_python.training_scripts.brax_infoprop_nopmap_train import main as infoprop_nopmap_main

@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig):
    if cfg.video_eval:
        from wheelbot_sim_python.eval_scripts.video_eval import run as video_eval_run
        video_eval_run(cfg.eval)
    elif cfg.algorithm.name == 'infoprop':
        infoprop_main(cfg)
    elif cfg.algorithm.name == 'infoprop_nopmap':
        infoprop_nopmap_main(cfg)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")


if __name__ == "__main__":
    main()
