# COMMANDS.md

## 1. 激活环境
```bash
conda activate shiprl

测试最基础船舶环境
python scripts/test_ship_env.py

测试随机 rollout
python scripts/test_ship_env_random.py

测试 gym 风格环境
python scripts/test_ship_env_gymstyle.py

测试 MARL wrapper rollout
python scripts/test_marl_rollout.py

测试最小策略网络交互
python scripts/test_policy_interaction.py

测试 actor-critic 训练骨架
python scripts/test_training_skeleton.py

测试最小 PPO-style 更新
python scripts/test_ppo_update.py

随机策略 baseline
python scripts/eval_random_policy.py

规则策略 baseline v1
python scripts/eval_rule_policy.py

规则策略 baseline v2
python scripts/eval_rule_policy_v2.py

baseline 对比
python scripts/compare_baselines.py

自己写的最小 RL 训练v1 
python scripts/train_minimal_rl.py

自己写的最小 RL 训练v2
python scripts/train_minimal_rl_v2.py

自己写的最小 RL 训练v3
python scripts/train_minimal_rl_v3.py

自己写的最小 RL 训练v4
python scripts/train_minimal_rl_v4.py

诊断自己写的最小 RL / MAPPO-like 结果
python scripts/diagnose_minimal_rl.py

on-policy 仓库位置
D:\ship_marl_project\external\on-policy

进入仓库
cd /d D:\ship_marl_project\external\on-policy

安装 on-policy 仓库
pip install -e .

测试是否安装成功
python -c "import onpolicy; print('onpolicy import success')"

官方 MPE MAPPO demo（已经能跑通）
python onpolicy\scripts\train\train_mpe.py --env_name MPE --algorithm_name mappo --experiment_name debug_mpe --scenario_name simple_spread --num_agents 3 --num_landmarks 3 --seed 1 --n_rollout_threads 1 --n_training_threads 1 --n_eval_rollout_threads 1 --num_mini_batch 1 --episode_length 25 --num_env_steps 10000 --ppo_epoch 5 --use_eval --eval_interval 10 --eval_episodes 4 --log_interval 1 --save_interval 10 --use_wandb --cuda

Ship + MLP-MAPPO 训练（已经能跑通）
python onpolicy\scripts\train\train_ship.py --env_name Ship --algorithm_name mappo --experiment_name debug_ship --scenario_name head_on --num_agents 2 --seed 1 --n_rollout_threads 1 --n_training_threads 1 --n_eval_rollout_threads 1 --num_mini_batch 1 --episode_length 50 --num_env_steps 10000 --ppo_epoch 5 --use_eval --eval_interval 10 --eval_episodes 4 --log_interval 1 --save_interval 10 --use_wandb --cuda

Ship + RMAPPO 训练（下一步重点）
python onpolicy\scripts\train\train_ship.py --env_name Ship --algorithm_name rmappo --experiment_name debug_ship_rmappo --scenario_name head_on --num_agents 2 --seed 1 --n_rollout_threads 1 --n_training_threads 1 --n_eval_rollout_threads 1 --num_mini_batch 1 --episode_length 50 --num_env_steps 10000 --ppo_epoch 5 --use_eval --eval_interval 10 --eval_episodes 4 --log_interval 1 --save_interval 10 --use_wandb --cuda

分析 MLP-MAPPO 训练日志
python scripts/analyze_ship_mappo_logs.py

诊断训练好的 MLP-MAPPO 策略
python onpolicy\scripts\diagnose_ship_mappo_v2.py --env_name Ship --algorithm_name mappo --experiment_name debug_ship --scenario_name head_on --num_agents 2 --episode_length 50 --n_rollout_threads 1 --n_training_threads 1 --cuda --use_wandb --model_dir D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\mappo\debug_ship\run4\models

关键结果路径：
MLP-MAPPO 结果目录
D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\mappo\debug_ship\run4

MLP-MAPPO 模型目录
D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\mappo\debug_ship\run4\models

MLP-MAPPO 日志文件
D:\ship_marl_project\external\on-policy\onpolicy\scripts\results\Ship\head_on\mappo\debug_ship\run4\logs\summary.json

日志分析图保存目录
D:\ship_marl_project\figures\mappo_analysis


---

# 二、`CURRENT_FINDINGS.md`

```md
# CURRENT_FINDINGS.md

## Project Topic
Multi-ship collision avoidance using multi-agent reinforcement learning, with the long-term goal of building a high-quality paper.

---

## 1. Environment status
The custom ship environment is already built and verified:
- `MultiShipEnv` works
- reward, collision detection, goal checking, render all work
- MARL wrapper works
- gym-style interface works

Current task setting:
- 2 ships
- head_on scenario mainly used for current debugging/training
- simplified kinematics
- collision avoidance + goal reaching

---

## 2. Heuristic baseline findings
### Random policy
- usually low success
- often high timeout
- collision not always high because policy is chaotic and inefficient

### Rule baseline v1
- simple distance-triggered right-turn rule
- often worse than random in collision rate
- too naive

### Rule baseline v2
- added relative bearing awareness and danger sector
- improved over rule v1
- but still not enough
- still failed to solve the task robustly

Conclusion:
Heuristic rules are insufficient for this task.

---

## 3. Self-written RL findings
### Minimal PPO-like MLP baseline
- training pipeline works
- actor/critic/buffer/update all verified
- but policy converges to a local optimum:
  - collision avoided
  - timeout dominates
  - goal not reached
  - excessive turning / looping

### Reward shaping experiments (v2 / v3 / v4)
- reward modifications changed behavior
- but did not fundamentally solve the local optimum
- policy still tends toward “safe but timeout”

### GRU minimal RL baseline
- did not show clear improvement under self-written minimal training framework
- no stable success trend observed

Conclusion:
The self-written minimal RL framework is enough for diagnosis,
but not reliable enough for final main training.
The local optimum is likely related to representation and task difficulty, not just code bugs.

---

## 4. Mature MAPPO integration findings
The custom ship environment has been successfully integrated into the open-source `on-policy` MAPPO framework.

### MLP-MAPPO has been trained successfully
Training command works:
- Ship + MAPPO (MLP)
- results saved in run4

### Log analysis findings
- MAPPO shows more learning dynamics than the self-written minimal RL baseline
- average reward / eval reward change more visibly
- policy entropy decreases over training
- ratio remains stable around 1
- value loss fluctuates but training is not collapsed

Conclusion:
Mature MAPPO is working and more stable than the self-written minimal RL.

---

## 5. MLP-MAPPO diagnosis findings
Using `diagnose_ship_mappo_v2.py`, the learned MLP-MAPPO policy was diagnosed in a fixed head_on scenario.

### Key results
- final collision = False
- final timeout = True
- final reached = [False, False]

### Action statistics
- action counts: {0: 55, 1: 3, 2: 42}
- action 1 (keep straight) is rarely chosen
- the policy mostly alternates between left and right turning

### Trajectory behavior
- ships avoid collision
- but they do not return to goal
- trajectories show large detours / looping / over-avoidance
- still trapped in a “safe but inefficient” local optimum

Conclusion:
Even under a mature MAPPO implementation, the basic MLP policy still converges to a collision-free but timeout-dominated local optimum.

This is a very important research finding:
the problem is not only unstable training implementation,
but also insufficient representation capacity of the current MLP policy.

---

## 6. Main research conclusion so far
We have already established a strong experimental chain:

1. Random / heuristic baselines are insufficient
2. Self-written minimal RL also fails with safe-but-timeout local optimum
3. Mature MLP-MAPPO is more stable, but still fails in a similar way
4. Therefore, the next step should focus on temporal representation / recurrent modeling

This strongly motivates:
- recurrent MAPPO (rmappo)
- then later GNN / Transformer extensions

---

## 7. Most important current insight
The current bottleneck is:
- not environment bugs
- not basic training bugs
- but the inability of the current MLP policy to learn “avoid then return to goal”

This makes recurrent / temporal modeling a natural and justified next step.

长期论文主线
heuristic baseline 失败
简单 RL 陷入 safe-but-timeout 局部最优
成熟 MLP-MAPPO 更稳定但仍然不足
时序建模是必要的
图结构交互建模进一步提升表现
最终方法在更复杂、更真实场景下验证有效