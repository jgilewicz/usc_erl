# Evolutionary Reinforcement Learning - Statistical & Performance Report
This report contains performance results, statistical significance tests, critic correlation tables, and experimental plots for five MuJoCo environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, and Swimmer-v5.

---

## HalfCheetah-v5

### Performance Summary Table

| Algorithm/Method | Eval_Reward_Mean | Eval_Reward_Std | Best_Pop_Fitness_Mean | Best_Pop_Fitness_Std |
| --- | --- | --- | --- | --- |
| PPO (Baseline) | -457.266219 | 264.761374 | NaN | NaN |
| TD3 (Baseline) | 8382.025787 | 1407.009015 | NaN | NaN |
| DDPG (Baseline) | 3737.256380 | 1162.420948 | NaN | NaN |
| SC-ERL (Random) | 4612.196428 | NaN | 4801.561399 | NaN |
| SC-ERL (Ensemble) [Ours] | 6193.027118 | 4594.783802 | 674.815417 | 472.842095 |
| SC-ERL (Dropout) [Ours] | 6618.254675 | 4538.812551 | 3140.230074 | 5070.300896 |

### Statistical Significance Table

| Zestawienie | Shapiro p-value (A, B) | Wybrany Test (T-test / M-W) | Stat p-value | Istotność (Significance) |
| --- | --- | --- | --- | --- |
| SC-ERL (Ensemble) [Ours] vs PPO (Baseline) | (0.6912, 0.0218) | Mann-Whitney U | 0.0357 | * |
| SC-ERL (Ensemble) [Ours] vs TD3 (Baseline) | (0.6912, 0.5973) | Welch's t-test | 0.4380 | ns |
| SC-ERL (Ensemble) [Ours] vs DDPG (Baseline) | (0.6912, 0.6517) | Welch's t-test | 0.4656 | ns |
| SC-ERL (Dropout) [Ours] vs PPO (Baseline) | (0.1692, 0.0218) | Mann-Whitney U | 0.0159 | * |
| SC-ERL (Dropout) [Ours] vs TD3 (Baseline) | (0.1692, 0.5973) | Welch's t-test | 0.3790 | ns |
| SC-ERL (Dropout) [Ours] vs DDPG (Baseline) | (0.1692, 0.6517) | Welch's t-test | 0.3465 | ns |

### Critic Correlation Analysis

| Algorithm/Method | Pearson Correlation | Spearman Correlation | Sample Size (N) |
| --- | --- | --- | --- |
| Ensemble Seed 4 | 0.1278 | 0.7115 | 356 |
| Dropout Seed 4 | 0.6983 | 0.6685 | 364 |

### Performance & Analysis Plots


## Hopper-v5

### Performance Summary Table

| Algorithm/Method | Eval_Reward_Mean | Eval_Reward_Std | Best_Pop_Fitness_Mean | Best_Pop_Fitness_Std |
| --- | --- | --- | --- | --- |
| PPO (Baseline) | 389.400919 | 199.997395 | NaN | NaN |
| TD3 (Baseline) | 2258.527735 | 1340.911895 | NaN | NaN |
| DDPG (Baseline) | 1546.238482 | 770.732397 | NaN | NaN |
| ERL (Baseline) | 643.381071 | 254.548248 | 1822.226041 | 903.759741 |
| SC-ERL (Random) | 846.931389 | 518.822941 | 479.124863 | 730.940524 |
| SC-ERL (Ensemble) [Ours] | 383.250079 | 489.909211 | 351.172076 | 597.104666 |
| SC-ERL (Dropout) [Ours] | 1358.076308 | NaN | 242.471875 | NaN |

### Statistical Significance Table

| Zestawienie | Shapiro p-value (A, B) | Wybrany Test (T-test / M-W) | Stat p-value | Istotność (Significance) |
| --- | --- | --- | --- | --- |
| SC-ERL (Ensemble) [Ours] vs PPO (Baseline) | (0.6429, 0.7999) | Welch's t-test | 0.4847 | ns |
| SC-ERL (Ensemble) [Ours] vs TD3 (Baseline) | (0.6429, 0.2722) | Welch's t-test | 0.1234 | ns |
| SC-ERL (Ensemble) [Ours] vs DDPG (Baseline) | (0.6429, 0.0120) | Mann-Whitney U | 0.5714 | ns |
| SC-ERL (Ensemble) [Ours] vs ERL (Baseline) | (0.6429, 0.5433) | Welch's t-test | 0.8611 | ns |
| SC-ERL (Ensemble) [Ours] vs SC-ERL (Random) | (0.6429, 0.2162) | Welch's t-test | 0.7229 | ns |

### Critic Correlation Analysis

| Algorithm/Method | Pearson Correlation | Spearman Correlation | Sample Size (N) |
| --- | --- | --- | --- |
| Ensemble Seed 4 | 0.5763 | 0.6832 | 472 |

### Performance & Analysis Plots


## Walker2d-v5

### Performance Summary Table

| Algorithm/Method | Eval_Reward_Mean | Eval_Reward_Std | Best_Pop_Fitness_Mean | Best_Pop_Fitness_Std |
| --- | --- | --- | --- | --- |
| PPO (Baseline) | 141.849680 | 170.844515 | NaN | NaN |
| TD3 (Baseline) | 3295.335293 | 821.222571 | NaN | NaN |
| DDPG (Baseline) | 1419.053778 | 910.133764 | NaN | NaN |
| ERL (Baseline) | 624.797626 | 214.821580 | 1041.399753 | 448.272209 |
| SC-ERL (Random) | 13.596905 | 35.450147 | 1020.637244 | 46.620501 |
| SC-ERL (Ensemble) [Ours] | 50.413706 | 100.489582 | -102.366629 | 490.716323 |

### Statistical Significance Table

| Zestawienie | Shapiro p-value (A, B) | Wybrany Test (T-test / M-W) | Stat p-value | Istotność (Significance) |
| --- | --- | --- | --- | --- |
| SC-ERL (Ensemble) [Ours] vs PPO (Baseline) | (N/A, 0.4255) | Mann-Whitney U | 0.3810 | ns |
| SC-ERL (Ensemble) [Ours] vs TD3 (Baseline) | (N/A, 0.5126) | Mann-Whitney U | 0.0952 | ns |
| SC-ERL (Ensemble) [Ours] vs DDPG (Baseline) | (N/A, 0.0188) | Mann-Whitney U | 0.3810 | ns |
| SC-ERL (Ensemble) [Ours] vs ERL (Baseline) | (N/A, N/A) | Mann-Whitney U | 0.3333 | ns |
| SC-ERL (Ensemble) [Ours] vs SC-ERL (Random) | (N/A, N/A) | Mann-Whitney U | 0.3333 | ns |

### Critic Correlation Analysis

| Algorithm/Method | Pearson Correlation | Spearman Correlation | Sample Size (N) |
| --- | --- | --- | --- |
| Ensemble Seed 4 | 0.6587 | 0.8765 | 214 |

### Performance & Analysis Plots


## Ant-v5

### Performance Summary Table

| Algorithm/Method | Eval_Reward_Mean | Eval_Reward_Std | Best_Pop_Fitness_Mean | Best_Pop_Fitness_Std |
| --- | --- | --- | --- | --- |
| PPO (Baseline) | -2730.744882 | 350.497725 | NaN | NaN |
| TD3 (Baseline) | 834.512101 | 906.162763 | NaN | NaN |
| DDPG (Baseline) | -1132.072426 | 677.384460 | NaN | NaN |
| ERL (Baseline) | 117.474574 | 1449.412498 | 995.571793 | 5.550125 |
| SC-ERL (Random) | -95.694563 | 32.957034 | 994.435093 | 4.338825 |
| SC-ERL (Ensemble) [Ours] | -209.975494 | 1281.069797 | 1.317865 | 96.987846 |
| SC-ERL (Dropout) [Ours] | -84.187083 | 49.584339 | 137710.531666 | 194819.938444 |

### Statistical Significance Table

| Zestawienie | Shapiro p-value (A, B) | Wybrany Test (T-test / M-W) | Stat p-value | Istotność (Significance) |
| --- | --- | --- | --- | --- |
| SC-ERL (Ensemble) [Ours] vs PPO (Baseline) | (0.7255, 0.4328) | Welch's t-test | 0.0138 | * |
| SC-ERL (Ensemble) [Ours] vs TD3 (Baseline) | (0.7255, 0.5164) | Welch's t-test | 0.3174 | ns |
| SC-ERL (Ensemble) [Ours] vs DDPG (Baseline) | (0.7255, 0.3367) | Welch's t-test | 0.2274 | ns |
| SC-ERL (Ensemble) [Ours] vs ERL (Baseline) | (0.7255, 0.0162) | Mann-Whitney U | 0.8857 | ns |
| SC-ERL (Ensemble) [Ours] vs SC-ERL (Random) | (0.7255, 0.9400) | Welch's t-test | 0.5990 | ns |
| SC-ERL (Dropout) [Ours] vs PPO (Baseline) | (N/A, 0.4328) | Mann-Whitney U | 0.0952 | ns |
| SC-ERL (Dropout) [Ours] vs TD3 (Baseline) | (N/A, 0.5164) | Mann-Whitney U | 0.0952 | ns |
| SC-ERL (Dropout) [Ours] vs DDPG (Baseline) | (N/A, 0.3367) | Mann-Whitney U | 0.0952 | ns |
| SC-ERL (Dropout) [Ours] vs ERL (Baseline) | (N/A, 0.0162) | Mann-Whitney U | 0.5333 | ns |
| SC-ERL (Dropout) [Ours] vs SC-ERL (Random) | (N/A, 0.9400) | Mann-Whitney U | 0.4000 | ns |

### Critic Correlation Analysis

| Algorithm/Method | Pearson Correlation | Spearman Correlation | Sample Size (N) |
| --- | --- | --- | --- |
| Ensemble Seed 4 | 0.4769 | 0.5428 | 321 |
| Ensemble Seed 3 | 0.3687 | 0.4494 | 432 |
| Dropout Seed 4 | 0.7337 | 0.9772 | 1339 |

### Performance & Analysis Plots


## Swimmer-v5

### Performance Summary Table

| Algorithm/Method | Eval_Reward_Mean | Eval_Reward_Std | Best_Pop_Fitness_Mean | Best_Pop_Fitness_Std |
| --- | --- | --- | --- | --- |
| PPO (Baseline) | 63.865830 | 21.952593 | NaN | NaN |
| TD3 (Baseline) | 62.445357 | 13.977818 | NaN | NaN |
| DDPG (Baseline) | 77.558928 | 45.599112 | NaN | NaN |
| ERL (Baseline) | 40.800199 | 5.359329 | 243.513551 | 62.395849 |
| SC-ERL (Random) | 43.082992 | NaN | 218.125717 | NaN |
| SC-ERL (Ensemble) [Ours] | 79.330420 | NaN | 125.530044 | NaN |
| SC-ERL (Dropout) [Ours] | 88.787072 | 84.786225 | 58.883233 | 55.383145 |

### Statistical Significance Table

| Zestawienie | Shapiro p-value (A, B) | Wybrany Test (T-test / M-W) | Stat p-value | Istotność (Significance) |
| --- | --- | --- | --- | --- |
| SC-ERL (Dropout) [Ours] vs PPO (Baseline) | (N/A, 0.5748) | Mann-Whitney U | 1.0000 | ns |
| SC-ERL (Dropout) [Ours] vs TD3 (Baseline) | (N/A, 0.0205) | Mann-Whitney U | 1.0000 | ns |
| SC-ERL (Dropout) [Ours] vs DDPG (Baseline) | (N/A, 0.0332) | Mann-Whitney U | 0.8571 | ns |
| SC-ERL (Dropout) [Ours] vs ERL (Baseline) | (N/A, 0.6887) | Mann-Whitney U | 0.8000 | ns |

### Critic Correlation Analysis

### Performance & Analysis Plots


### Plots - HalfCheetah-v5
![Sample Efficiency - HalfCheetah-v5](./HalfCheetah-v5/HalfCheetah-v5_sample_efficiency.png)
![Surrogate Analysis - HalfCheetah-v5](./HalfCheetah-v5/HalfCheetah-v5_surrogate_analysis.png)
![Critic Correlation - HalfCheetah-v5](./HalfCheetah-v5/HalfCheetah-v5_critic_correlation.png)

---

### Plots - Hopper-v5
![Sample Efficiency - Hopper-v5](./Hopper-v5/Hopper-v5_sample_efficiency.png)
![Surrogate Analysis - Hopper-v5](./Hopper-v5/Hopper-v5_surrogate_analysis.png)
![Critic Correlation - Hopper-v5](./Hopper-v5/Hopper-v5_critic_correlation.png)

---

### Plots - Walker2d-v5
![Sample Efficiency - Walker2d-v5](./Walker2d-v5/Walker2d-v5_sample_efficiency.png)
![Surrogate Analysis - Walker2d-v5](./Walker2d-v5/Walker2d-v5_surrogate_analysis.png)
![Critic Correlation - Walker2d-v5](./Walker2d-v5/Walker2d-v5_critic_correlation.png)

---

### Plots - Ant-v5
![Sample Efficiency - Ant-v5](./Ant-v5/Ant-v5_sample_efficiency.png)
![Surrogate Analysis - Ant-v5](./Ant-v5/Ant-v5_surrogate_analysis.png)
![Critic Correlation - Ant-v5](./Ant-v5/Ant-v5_critic_correlation.png)

---

### Plots - Swimmer-v5
![Sample Efficiency - Swimmer-v5](./Swimmer-v5/Swimmer-v5_sample_efficiency.png)
![Surrogate Analysis - Swimmer-v5](./Swimmer-v5/Swimmer-v5_surrogate_analysis.png)
![Critic Correlation - Swimmer-v5](./Swimmer-v5/Swimmer-v5_critic_correlation.png)

---
