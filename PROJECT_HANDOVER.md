# Ship MARL Project Handover

## 1. Project Goal
Develop a high-quality paper on multi-ship collision avoidance using multi-agent reinforcement learning.
Current strategy:
- Start from minimal environment
- Build baselines
- Verify MAPPO
- Then move to recurrent / graph / transformer extensions

---

## 2. Local Environment
- OS: Windows
- Python env: conda env `shiprl`
- Main IDE: VSCode
- Project root:
  `D:\ship_marl_project`

---

## 3. My Own Project Code Status

### Environment
Implemented and runnable:
- `envs/multi_ship_env.py`
- ship dynamics, reward, collision detection, goal checking, rendering
- current reward version: reward v4

### Wrappers / utilities implemented
- `envs/multi_ship_marl_wrapper.py`
- random baseline eval
- rule baseline v1
- rule baseline v2
- baseline comparison script
- minimal PPO-style training skeleton
- minimal RL v1/v2/v3/v4 experiments
- GRU minimal RL experiment
- diagnosis scripts for minimal RL

### Key diagnosis result before MAPPO
Both minimal RL and GRU baseline tend to converge to a "safe but timeout" local optimum:
- collision=False
- timeout=True
- reached=False
- behavior = excessive turning / looping / no return to goal

---

## 4. on-policy MAPPO Integration Status

### External repo
Location:
`D:\ship_marl_project\external\on-policy`

Successfully done:
- `pip install -e .`
- official MPE MAPPO demo runs successfully
- ship environment has been adapted to on-policy

### Added files in on-policy
- `onpolicy/envs/ship/__init__.py`
- `onpolicy/envs/ship/Ship_env.py`
- `onpolicy/scripts/train/train_ship.py`
- `onpolicy/scripts/diagnose_ship_mappo.py`
- `onpolicy/scripts/diagnose_ship_mappo_v2.py`

### Important compatibility modifications
- `onpolicy/runner/shared/mpe_runner.py`
  modified logging block so that `env_infos` is always defined, not only for env_name == "MPE"

### ShipEnv design
- Discrete action adapter for underlying `MultiShipEnv`
- action mapping:
  - 0 = left
  - 1 = keep
  - 2 = right
- local obs dim = 10
- share obs dim = 20
- reward returned as shape `(n_agents, 1)` to match on-policy shared buffer

---

## 5. MLP-MAPPO Baseline Status

### Training command that runs successfully
```bash
python onpolicy\scripts\train\train_ship.py --env_name Ship --algorithm_name mappo --experiment_name debug_ship --scenario_name head_on --num_agents 2 --seed 1 --n_rollout_threads 1 --n_training_threads 1 --n_eval_rollout_threads 1 --num_mini_batch 1 --episode_length 50 --num_env_steps 10000 --ppo_epoch 5 --use_eval --eval_interval 10 --eval_episodes 4 --log_interval 1 --save_interval 10 --use_wandb --cuda