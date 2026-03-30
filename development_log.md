# Development Log
<!-- 日志 -->

## Milestone 1
- Built the first runnable multi-ship environment
- Implemented reset(), step(), reward, collision detection, goal checking, and render()
- Verified that head-on straight-line motion leads to collision
- Verified that trajectories are physically intuitive
- 2D平面海域，每艘船状态包括位置、航向、速度。采用圆形碰撞域。目标到达阈值定义成功。
- 场景库原型：对遇、交叉、追越
- 奖励机制初版：goal progress、collision penalty、step penalty、smooth penalty

## Milestone 2: Training-ready environment
- Added random perturbations to initial positions and headings
- Added obs_dim and action_dim
- Added seed() and close()
- Verified multi-episode random rollout stability

## Milestone 3: Gym-style environment
- Added observation_space and action_space
- Added terminated / truncated flags
- Added global state interface for future centralized critic
- Verified compatibility with Gym-style interaction

## Milestone 4: Random policy evaluation pipeline
- Implemented batch evaluation script for random policy
- Collected success rate, collision rate, timeout rate, average return, and average episode length
- Established the first unified experiment protocol

## Milestone 5: Initial rule-based baseline
- Implemented a simple reactive rule-based baseline combining goal-following and distance-triggered right-turn avoidance
- Evaluated the rule policy under head-on, crossing, and overtaking scenarios
- Observed that the naive rule policy did not outperform the random policy in head-on and crossing scenarios
- This suggests that simple distance-triggered heuristics are insufficient for handling interaction geometry in multi-ship encounter tasks

## Milestone 6: Improved heuristic baseline
- Designed an improved rule-based baseline with relative-bearing awareness and forward danger sector detection
- Observed clear improvements over the naive rule policy in head-on and crossing scenarios
- However, the improved heuristic baseline still failed to outperform the random policy in collision rate, while reducing timeout
- This suggests that heuristic reactive rules are insufficient to jointly optimize safety and task efficiency, motivating the use of learning-based decision models

## Milestone 7: Baseline comparison
- Merged summary results of Random, Rule_v1, and Rule_v2 baselines
- Generated baseline comparison CSV and metric bar plots
- Observed that Rule_v2 improves over Rule_v1 in collision handling, while Random still shows lower collision rates due to high timeout behavior
- Established the first baseline comparison pipeline for future RL experiments

## Milestone 8: Minimal PPO-style update verification
- Implemented simple discounted return computation
- Implemented advantage computation and normalization
- Flattened MARL rollout data into training batches
- Verified actor loss, critic loss, entropy, backward propagation, and optimizer update
- Confirmed that both actor and critic parameters are updated successfully