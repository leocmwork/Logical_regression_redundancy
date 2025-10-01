# Logistic Regression under Redundancy

Program generates binary-classification datasets with varying levels of feature correlation and compares Ridge, Elastic Net, and Lasso logistic regression using ROC AUC.

---

## Model details

**True predictors**  

(five independent standard-normal features)

$$
Z_1, \ldots, Z_5 \sim \mathcal N(0,1) \quad \text{(i.i.d.)}
$$

$$
Z = (Z_1, \ldots, Z_5)
$$


**Redundant predictors**

(clones of $Z_1$ with target correlation $r$)

$$
X = r Z_1 + \sqrt{1-r^2}\cdot E, 
\qquad 
E \sim \mathcal N(0,1)
$$


(mixtures of $Z_2$ and $Z_3$ (weights $\alpha,\beta$))

$$
X = \alpha Z_2 + \beta Z_3 + \gamma E, 
\qquad 
\gamma = \sqrt{\max\bigl(0\,1-\alpha^2-\beta^2\bigr)}
$$

(one redundant feature is added per dataset)

**Target**

$$
\eta = \beta^\top Z + \epsilon
\qquad 
\epsilon \sim \mathcal N(0,\sigma^2)
\qquad 
p  = \frac{1}{1+\exp(-\eta)}
\qquad 
y \sim \mathrm{Bernoulli}(p)
$$

(redundant features do not enter $\eta$ directly, they are predictive only via correlation with $Z$)

---

## Logistic regression with Elastic Net

**Minimize**

$$
\frac{1}{n} \sum_i \Big[ \log(1 + \exp(z_i)) - y_i \ z_i \Big]
\+\
\lambda \Big[ \tfrac{1-\rho}{2} \|w\|_2^2 + \rho \|w\|_1 \Big]
$$

$z_i = w^\top x_i + b$

$\rho \in [0,1]$ ($\rho=0$ Ridge, $\rho=1$ Lasso)

$\lambda \propto 1/C$

---

## Optimization

SAGA (stochastic variance-reduced gradient with proximal step for the L1 part)  

**Gradient**

$$
\nabla_w = (\sigma(z) - y) x
\qquad
\nabla_b = \sigma(z) - y
$$

**Proximal step**

$$
w_j \leftarrow\ \text{sign}(w_j)\max\bigl(|w_j| - \eta \lambda \rho\, 0 \bigr).
$$

**Hyperparameters** 

- $p \in \{0, 0.1, \dots, 1.0\}$  
- Run 5-fold CV over $C \in \{0.01, 0.1, 1, 10, 100\}$  
- Pick best $C$, refit on full train  

---

## Evaluation  

ROC AUC on test dataset  

$$
\text{AUC} = P(\text{score(positive)} > \text{score(negative)})
$$

---

## Parameters

n-samples: larger values give more stable estimates (lower variance in CV/test AUC)

beta: strength/sign of the true effects (larger magnitude gives higher achievable AUC)

noise-sd: randomness on log-odds (larger values lower the AUC ceiling; more realistic)

r-list / --mix-list: redundancy patterns

random-state: reproducibility

---

## Example (Windows) 

```bash

git clone https://github.com/leocmwork/Logical_regression_redundancy.git

cd Logical_regression_redundancy

python -m venv .venv

.\.venv\Scripts\Activate.ps1 

python.exe -m pip install --upgrade pip

pip install -r requirements.txt

python generate_redundancy_series.py --n-samples 300 --beta "1.2,-1.0,0.8,0.6,0.4" --noise-sd 0.01 --r-list "0.9,0.8,0.7,0.6,0.5" --mix-list "0.9,0.1;0.8,0.2;0.7,0.3;0.6,0.4;0.5,0.5" --random-state 42

python split_redundancy_series.py --glob "data/raw/redundancy_series_rank*.csv" --target target

python train_en_sweep_redundancy.py --glob "data/train/redundancy_series_rank*_train.csv" --target target

python evaluate_models_redundancy.py --glob "data/test/redundancy_series_rank*_test.csv" --target target

```

---

## Example (Linux/macOS)

```bash

git clone https://github.com/leocmwork/Logical_regression_redundancy.git

cd Logical_regression_redundancy

python -m venv .venv

.venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
  
python generate_redundancy_series.py \
  --n-samples 300 \
  --beta "1.2,-1.0,0.8,0.6,0.4" \
  --noise-sd 0.01 \
  --r-list "0.9,0.8,0.7,0.6,0.5" \
  --mix-list "0.9,0.1;0.8,0.2;0.7,0.3;0.6,0.4;0.5,0.5" \
  --random-state 42

python split_redundancy_series.py --glob "data/raw/redundancy_series_rank*.csv" --target target

python train_en_sweep_redundancy.py --glob "data/train/redundancy_series_rank*_train.csv" --target target

python evaluate_models_redundancy.py --glob "data/test/redundancy_series_rank*_test.csv" --target target

```

---

## Example results

![ROC AUC vs redundant predictors](https://private-user-images.githubusercontent.com/234937055/495882167-c6785085-b3ee-48c5-b204-b0ac4108ec80.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTkyNjQ1NjIsIm5iZiI6MTc1OTI2NDI2MiwicGF0aCI6Ii8yMzQ5MzcwNTUvNDk1ODgyMTY3LWM2Nzg1MDg1LWIzZWUtNDhjNS1iMjA0LWIwYWM0MTA4ZWM4MC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwOTMwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDkzMFQyMDMxMDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZTYxYjEyYjkyNDU5YzVmYzEyYTI0ZGUzMzljNTEwY2I5YzMzMTI2ZjBiYjQ0MTU5MDIxMDBiNzZmOGIxNjQ1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.umMhrb3qhbU1bO3dJTJGEm3Wi31AupKF2KcJbfrIDXg)



