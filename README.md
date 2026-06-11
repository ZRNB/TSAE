# Teacher-Student Active Exploration Framework for Retention Recommendation (TSAE)

## 简介 (Introduction)
This repository implements **TSAE (Teacher-Student Active Exploration Framework)**, a generative flow network (GFlowNets) based recommendation framework for **user retention optimization** in recommender systems.

User retention is a core long-term optimization goal for recommendation platforms, but it faces severe challenges: **cross-session delay, data sparsity, and weak supervision**. Existing retention-oriented recommendation methods mostly rely on implicit exploration, which easily falls into exploration blind spots and fails to fully mine high-value recommendation regions beneficial to long-term user revisit.

To address the above issues, this work proposes a teacher-student collaborative active exploration framework:
1. Decouple **user preference learning** and **exploration control** via teacher-student architecture.
2. The student model learns recommendation policy by fusing session-level immediate feedback and cross-session delayed retention signals.
3. The teacher model guides active exploration towards under-explored but high-retention-value regions based on the student's training discrepancy.
4. Adopt alternating sampling and shared replay buffer to realize efficient knowledge transfer between teacher and student.

Experimental results on multiple public datasets prove that TSAE significantly outperforms state-of-the-art baselines on both long-term retention metrics and short-term user interaction metrics.

## 论文信息 (Paper Info)
- **Title**: Teacher-Student Active Exploration Framework for Retention Recommendation
- **Conference**: ICDE
- **Authors**: Rui Zhu, Yudong Zhang*, Feng Niu, Xuan Yu, Xu Wang, Yang Wang*
- **Affiliation**: University of Science and Technology of China
- **Corresponding Authors**: Yudong Zhang, Yang Wang

## 方法框架 (Method Overview)
### 核心架构
The framework consists of two core modules: **Student Model** and **Teacher Model**, sharing the user state encoder, and cooperating through alternating sampling & shared replay buffer.
1. **Shared User State Encoder**
   Built with Transformer, encodes user profile and historical interaction sequences into unified user state embeddings for both student and teacher.

2. **Student Model (Main Recommendation Policy)**
   - Formulate list recommendation as a sequential trajectory generation task based on GFlowNets.
   - Design a unified terminal utility to fuse **immediate feedback (click, like, watch time, etc.)** and **delayed user retention signal**.
   - Optimize via refined Detailed Balance (DB) loss, realizing step-wise supervision from immediate feedback and terminal constraint from retention signal.

3. **Teacher Model (Exploration Guidance)**
   - Take the student's DB discrepancy as the core exploration reward, focusing on trajectories that the student under-fits.
   - Combine retention utility to avoid meaningless exploration, and learn an exploration-oriented generative policy.

4. **Alternating Sampling Mechanism**
   Collect trajectories alternately by student and teacher, store all samples into a shared replay buffer. The student learns from both self-generated and teacher-guided exploration trajectories to eliminate exploration blind spots.

### 整体训练流程
1. Encode user request to get user state.
2. Alternately use student/teacher policy to generate recommendation lists (trajectories).
3. Collect immediate rewards and delayed retention signals, store trajectories into replay buffer.
4. Sample mini-batch to update student and teacher parameters respectively.
5. Iterate until model convergence; **only the student model is used for online inference**.

## 数据集 (Datasets)
The experiments adopt three real-world public datasets:

| Dataset | # Users | # Items | # Interactions | Density | Description |
|---------|---------|---------|----------------|---------|-------------|
| KuaiRand-Pure | 27,285 | 7,551 | 1,436,609 | 0.70% | Unbiased short-video recommendation dataset |
| KuaiRand-27K | 27,285 | 507,493 | 71,487,342 | 0.05% | Full-scale KuaiRand dataset with massive interactions |
| MovieLens-1M | 6,400 | 3,706 | 1,000,208 | 4.22% | Classic dense movie rating dataset |

- User Simulator: Use **KuaiSim** to simulate user session behavior and next-day revisit for offline evaluation.
- Feedback Definition:
  1. KuaiRand: 6 positive feedbacks (click, view, like, comment, follow, forward) + 2 negative feedbacks (hate, leave).
  2. MovieLens-1M: Binarize ratings (score > 3 as positive, others as negative).

## 环境依赖 (Requirements)
```txt
python >= 3.8
pytorch >= 1.9.0
numpy
pandas
scikit-learn
tqdm


实验结果 (Experimental Results)
评价指标
Long-term Retention Metrics: Return Time (↓), Retention (↑)
Short-term Interaction Metrics: Click Rate (↑), Long View Rate (↑), Like Rate (↑)
整体性能
TSAE achieves SOTA performance across all datasets, outperforming TD3, SAC, DIN, CEM, RLUR, GFN4Retention on all metrics:
Reduce user return time by more than 40% compared with baseline TD3.
Improve retention by over 12% consistently.
Obtain significant gains on click rate, long view rate and like rate without sacrificing short-term experience.
消融实验结论
Unified Reward: Fusing immediate feedback and retention signal is necessary; removing either component leads to obvious performance drop.
Encoder Structure: Shared user encoder performs better than fully independent encoders for teacher-student collaboration.
Sampling Ratio: Sampling ratio = 1.0 (alternate sampling) achieves the best balance of exploration and exploitation.
Discrepancy Strategy: Linear discrepancy enhancement is more stable than non-linear power amplification.
计算开销
TSAE only brings 9%~12% extra GPU memory overhead and moderate training time increase, with acceptable computational cost for industrial deployment.
主要贡献 (Contributions)
Propose a teacher-guided GFlowNets framework (TSAE) for retention recommendation, solving the defect of implicit exploration in existing methods.
Design teacher-student alternating sampling & collaborative training mechanism to realize directional active exploration.
Construct a unified modeling scheme for immediate session feedback and cross-session retention signal, supporting flexible multi-objective optimization.
Sufficient experiments and ablation studies verify the effectiveness, robustness and efficiency of the proposed method on multiple datasets.