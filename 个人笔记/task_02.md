### task02 grpo demo

#### 零、前言
关于GRPO的内容，可以参考我的个人网页，包括[强化学习笔记](http://127.0.0.1:5173/rl-book/cha6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0#grpo-group-relative-policy-optimization)、[deepseek-r1论文解读](http://127.0.0.1:5173/papers/deepseek-r1/)

**GRPO(组相对策略优化算法)** 的核心是：对于同一个 prompt 一次采样多个回答，让这些回答在组内相互比较，用“相对于组内其他回答好多少”来构造优势，而不再像 PPO 那样额外训练一个 Critic / Value Model 来估计价值函数。

相较于PPO，最大的优势是极大地降低了训练过程的显存占用，提高了训练的效率。

RLVR，可验证奖励的特点是根据规则来确定奖励分数，常用于数学/代码等验证性强的场景。

GSM8k是由8500道数学问题组成（**需要注意，基本是一些考察最基本数学运算模式的题目**，因此4B模型正确率可能本来就不低）的benchmark数据集。

#### 一、代码解读

给出如下架构图
![](figures/grpo-demo-architecture.png)

#### 二、运行结果

##### 2.1 三种损失函数的对比

浅色点保留逐步观测值，实线从第 9 步开始，表示含当前步在内的最近 10 步均值。loss 的均值只计入窗口内实际更新的步骤。

![平均奖励：IS、PPO、CISPO](figures/task02/grpo-demo-compare-reward.png)

100 步的平均训练奖励分别为 IS 0.872、PPO 0.871、CISPO 0.873，最高与最低只差 0.002。三组在第 57 步附近都短暂下探，又很快回升。每步只采样 4 道题，单步均值容易受这一批题目和回答影响。

![平均生成长度（token）：IS、PPO、CISPO](figures/task02/grpo-demo-compare-avg_gen_len.png)

生成长度拉开了更明显的差距。最后 20 步，IS、PPO、CISPO 的平均生成长度分别为 266.7、140.6、247.1 个 token。PPO 这次用更短的回答取得了相近的训练奖励。三种损失都没有显式的长度惩罚；要把缩短归因于 PPO 裁剪，还需要按回答长度比较优势和实际触发裁剪的 token 比例。

![退化组比例：IS、PPO、CISPO](figures/task02/grpo-demo-compare-frac_degenerate.png)

三组都出现过整批退化：同一道题的 8 个回答得分相同，组内优势全部为零。100 步中，IS 和 PPO 各有 29 步整批退化，CISPO 有 39 步。因此，即使平均奖励接近，三组实际执行更新的次数也有差异。

![训练样本数：IS、PPO、CISPO](figures/task02/grpo-demo-compare-datums.png)

训练样本数与上一张图逐步对应。若退化组比例为 $f$，每步保留的回答数就是 $4 \times 8 \times (1-f)$。例如，退化比例为 0.75 时，只有一道题的 8 个回答进入训练。CISPO 多出的 10 个整批退化步骤在这张图上都对应零样本。

![训练 token 数：IS、PPO、CISPO](figures/task02/grpo-demo-compare-train_tokens.png)

训练 token 数同时取决于保留了多少条回答，以及这些回答有多长。读 PPO 后期的低值时，可以对照前两项指标：样本数相同的两步，也可能因为有效回答长度不同而有不同的训练量。生成长度图统计全部回答，训练 token 图只统计参与更新的部分。

![训练损失：IS、PPO、CISPO](figures/task02/grpo-demo-compare-loss.png)

设 $r$ 为某个 token 在当前策略与采样策略下的概率比，$A$ 为其组内优势。IS 用 $rA$ 构造目标，更新权重随 $r$ 变化。PPO 在正优势 token 的概率比超过上界、或负优势 token 的概率比低于下界时，裁剪分支会使该项梯度归零。CISPO 把裁剪后的 $r$ 当作固定权重，乘以 $A$ 和当前策略的对数概率；即使权重达到上界，对数概率仍能传递梯度。

这也解释了为何三条 loss 曲线的高低不能直接给方法排名：CISPO 记录的目标值显式含有对数概率。各更新步的平均概率比在 0.99885～1.00104 之间，裁剪前后的均值曲线重合。此次 CISPO 的裁剪区间为 $[0,4]$，61 个更新步记录的 `cispo/clip_fraction` 全为 0，说明参与更新的 token 都没有触发裁剪。在这个条件下，CISPO 与 IS 对单个 token 的策略梯度都与 $rA\,\nabla\log\pi$ 成正比。两组的生成长度走势相近，与这一性质相容；PPO 这次的裁剪比例没有记录，长度差异的成因仍待验证。

##### 2.2 With importance sampling loss

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 100 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

![GRPO demo IS · reward](figures/task02/grpo-demo-is-reward.png)

![GRPO demo IS · frac_degenerate](figures/task02/grpo-demo-is-frac_degenerate.png)

![GRPO demo IS · datums](figures/task02/grpo-demo-is-datums.png)

![GRPO demo IS · train_tokens](figures/task02/grpo-demo-is-train_tokens.png)

![GRPO demo IS · loss](figures/task02/grpo-demo-is-loss.png)

![GRPO demo IS · avg_gen_len](figures/task02/grpo-demo-is-avg_gen_len.png)

##### 2.3 With PPO loss

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 100 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn ppo \
    --swanlab-mode online
```

![GRPO demo PPO · reward](figures/task02/grpo-demo-ppo-reward.png)

![GRPO demo PPO · frac_degenerate](figures/task02/grpo-demo-ppo-frac_degenerate.png)

![GRPO demo PPO · datums](figures/task02/grpo-demo-ppo-datums.png)

![GRPO demo PPO · train_tokens](figures/task02/grpo-demo-ppo-train_tokens.png)

![GRPO demo PPO · loss](figures/task02/grpo-demo-ppo-loss.png)

![GRPO demo PPO · avg_gen_len](figures/task02/grpo-demo-ppo-avg_gen_len.png)

##### 2.4 With CISPO loss

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 100 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn cispo \
    --swanlab-mode online
```

![GRPO demo CISPO · reward](figures/task02/grpo-demo-cispo-reward.png)

![GRPO demo CISPO · frac_degenerate](figures/task02/grpo-demo-cispo-frac_degenerate.png)

![GRPO demo CISPO · datums](figures/task02/grpo-demo-cispo-datums.png)

![GRPO demo CISPO · train_tokens](figures/task02/grpo-demo-cispo-train_tokens.png)

![GRPO demo CISPO · loss](figures/task02/grpo-demo-cispo-loss.png)

![GRPO demo CISPO · avg_gen_len](figures/task02/grpo-demo-cispo-avg_gen_len.png)

![GRPO demo · cispo/train_tokens](figures/task02/grpo-demo-cispo-update-train_tokens.png)

![GRPO demo · cispo/clip_low_threshold](figures/task02/grpo-demo-cispo-clip_low_threshold.png)

![GRPO demo · cispo/clip_high_threshold](figures/task02/grpo-demo-cispo-clip_high_threshold.png)

![GRPO demo · cispo/ratio_mean](figures/task02/grpo-demo-cispo-ratio_mean.png)

![GRPO demo · cispo/clipped_ratio_mean](figures/task02/grpo-demo-cispo-clipped_ratio_mean.png)

![GRPO demo · cispo/clip_fraction](figures/task02/grpo-demo-cispo-clip_fraction.png)
