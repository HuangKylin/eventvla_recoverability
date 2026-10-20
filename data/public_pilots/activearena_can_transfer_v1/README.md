---
license: apache-2.0
task_categories:
- robotics
tags:
- active-perception
- astribot
- lerobot
pretty_name: ActiveArena Astribot trajectories
---

# ActiveArena-Data

Frozen LeRobot-format trajectories for the ActiveArena Astribot benchmark.
The repository contains 35 task directories, each with 100 training episodes
and the corresponding metadata files. The directory names match the
`activearena_astribot` registry in ActiveArena-VLA.

Use the files with the `ActiveArena_Astribot_lerobot` data root expected by the
released training configurations. This dataset snapshot is kept independent
of upstream simulator updates; the task and environment source revision is
recorded in the ActiveArena code repository's `docs/UPSTREAM_SNAPSHOTS.md`.

## Citation

If you use ActiveArena in your research, please cite the following paper.

```bibtex
@article{li2026activearena,
  title={ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation},
  author={Li, Yibo and Zhou, Enshen and Chen, Rui and Ding, Yanjun and Liu, Mengzhen and Han, Yi and Zhan, Jiabo and Wang, Lipeng and Zhang, Shanghang and Sheng, Lu},
  journal={arXiv preprint arXiv:2609.24124},
  year={2026}
}
```
