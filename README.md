# Customer Segmentation — E-Commerce Retail

Unsupervised machine learning project to segment customers based on their browsing and purchasing behavior. Built with a real e-commerce dataset, inspired by retail chains like Bravo and Araz.

## Overview

The goal is to group ~100,000 customers into meaningful segments so marketing teams can target each group differently — instead of sending the same message to everyone.

The model uses behavioral features beyond basic RFM, including how focused a customer's category browsing is, how efficiently they convert sessions into purchases, and whether they tend to shop on weekends.

## Dataset

[E-Commerce Behavior Data from Multi-Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) — October 2019

- ~2.5M rows after sampling 100,000 unique users
- Events: `view`, `cart`, `purchase`
- Columns: event_time, event_type, product_id, category_code, brand, price, user_id, user_session

## Features Used for Clustering

| Feature | Description |
|---|---|
| `recency` | Days since last event |
| `frequency` | Number of purchases |
| `monetary` | Total amount spent |
| `avg_price` | Average price of viewed/purchased items |
| `session_efficiency` | Purchases per session |
| `category_concentration` | HHI score — how focused the customer is on one category |
| `weekend_ratio` | Share of activity happening on weekends |

## Segments Found

| Cluster | Label | Description |
|---|---|---|
| 0 | Churn Risk | Low activity, no purchases, haven't engaged recently |
| 1 | VIP | Active, frequent buyers with high spend |
| 2 | Regular | Some purchases, average price range |
| 3 | Premium Occasional | High avg price but rare purchases |

## Method

- Sampled 100,000 unique users who had at least 5 events
- Built user-level features from all event types (view + cart + purchase)
- Applied `RobustScaler` to handle outliers
- Used Elbow method + Silhouette Score to pick optimal k
- K-Means with k=4, n_init=20
- Visualized clusters with PCA (2D)

## Results

Clusters are statistically distinct across all main features. The VIP segment (cluster 1) is small but drives a disproportionate share of revenue — a classic Pareto pattern.

## How to Run

```bash
pip install kagglehub scikit-learn pandas numpy matplotlib seaborn scipy
```

Open `Customer_Segmentation.ipynb` in Google Colab or Jupyter. The dataset downloads automatically via `kagglehub`.

## Libraries

- `scikit-learn` — KMeans, PCA, RobustScaler, silhouette_score
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualization
- `scipy` — Kruskal-Wallis statistical test
