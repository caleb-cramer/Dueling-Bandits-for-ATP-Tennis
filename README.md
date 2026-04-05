# Contextual Dueling Bandits: Optimizing Player Dominance in ATP Tennis

**Final Project - CSE541** **Author:** Caleb Cramer  
**Date:** March 16, 2026

## Executive Summary
This project applies an online learning framework to historical ATP match data (2008–2023) from [Kaggle](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) to identify optimal tennis players across varying environmental contexts (court surfaces). By treating players as "arms" in a **Dueling Bandit** problem, the system converges on the best candidates using relative preference feedback rather than noisy absolute rewards.

## High-Signal Technical Challenges

### 1. Handling Condorcet Cycles
Real-world data often lacks a "Condorcet Winner"—a player who defeats every other opponent individually. On Grass and Hard courts, "rock-paper-scissors" cycles were identified (e.g., Federer > Nadal > Djokovic > Federer).
* **Solution**: Pivoted to the **Borda Winner** criterion, which identifies the player with the highest average performance across the entire field.
* **Benefit**: Guarantees a stable winner exists even in finite sets with cycles and reduces sample complexity compared to Condorcet-seeking algorithms.

### 2. Eliminating Data Leakage
Initial implementations suffered from "look-ahead bias" by using career-average ranks to predict individual matches.
* **Solution**: Developed a **Sequential Match Environment**.
* **Implementation**: Matches are stored in chronological queues for each player pair. The algorithm "pops" the next historical match, updating player rank and form data dynamically based only on information available *before* that specific match.

### 3. Addressing Non-Stationarity (The "Rookie" Problem)
The professional tour is non-stationary; new talent constantly enters the pool. I tested algorithm adaptability by introducing a "Next Generation" of players (e.g., Medvedev, Tsitsipas) mid-simulation to evaluate how models handle the sudden introduction of new arms.

---

## Evaluated Algorithms

### Borda-UCB (Context-Free)
* **Mechanism**: Maintains empirical Borda scores $\hat{b}_{i}$ and selects players using a standard Upper Confidence Bound: $U_{i}=\hat{b}_{i}+\alpha\sqrt{\ln t/N_{i}}$.
* **Outcome**: Robust in stable environments but requires a significant "acclimatization" period when new players are introduced, leading to higher cumulative regret.

### Pairwise LinUCB (Contextual)
* **Mechanism**: Models preference probability $P(i>j)$ as a linear function of anti-symmetric feature vectors $\phi_{i,j}$.
* **Features**: Rank differences, best-of-three/five indicators, surface mastery, and recent form.
* **Outcome**: Successfully leveraged context to predict the utility of new players before dueling them, resulting in faster adaptation and lower cumulative regret.

---

## Performance Results

### Cumulative Regret Analysis
The contextual model (LinUCB) significantly outperformed the context-free baseline, particularly during the "Next Gen" injection.

| Surface | Algorithm | Performance Observation |
| :--- | :--- | :--- |
| **Clay** | Both | Accurately identified Rafael Nadal as the dominant "King of Clay." |
| **Hard** | LinUCB | Adapted rapidly to early upsets by "Next Gen" players. |
| **General** | LinUCB | Demonstrated sub-linear regret growth by treating new players as known feature vectors. |

### Top Predicted Players (Ground Truth vs. Model)
| Surface | Actual Borda Winner (Truth) | LinUCB Prediction |
| :--- | :--- | :--- |
| **Grass** | Federer R. (0.602) | Federer R. (0.513) |
| **Clay** | Nadal R. (0.868) | Nadal R. (0.585) |
| **Hard** | Djokovic N. (0.787) | Djokovic N. (0.642) |

---

## Implementation Details
* **Languages/Tools**: Python, NumPy, Pandas, Matplotlib.
* **Math Foundations**: Ridge regression for weight estimation ($\hat{\theta}=A^{-1}b$) and Cumulative Strong Regret ($R_{T}$) for evaluation.

```python
# Core logic for updating statistics in Pairwise LinUCB
# A is initialized as an Identity matrix
A = A + x_t @ x_t.T # Covariance update
b = b + (r_t - 0.5) * x_t # Mean update
