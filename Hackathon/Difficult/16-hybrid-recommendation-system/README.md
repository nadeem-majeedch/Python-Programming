# Problem 16: Hybrid Recommendation System

## Domain
Recommendation Systems

## Problem Statement
You are building the recommendation engine for a new e-commerce platform. The platform has both established users with rating histories and new users who just signed up (cold-start). Similarly, some products are well-established while others are newly listed with no ratings. A pure collaborative filtering approach fails for cold-start scenarios, and a pure content-based approach lacks serendipity.

Your task is to design a hybrid recommendation system that combines collaborative filtering (matrix factorization) with content-based methods. The system must produce accurate rating predictions for known users while still providing reasonable recommendations for cold-start users and new items.

## Objectives
1. Achieve RMSE < 1.0 on held-out rating predictions
2. Provide meaningful recommendations for cold-start users (no history) using demographic and content information
3. Achieve precision@k > 0.6 for top-10 recommendations
4. Demonstrate diversity/serendipity in recommendations beyond simply popular items

## Dataset
- 500 users with demographics (age, region)
- 200 items with metadata (category, price, popularity)
- 10000 interactions (ratings 1-5)
- Includes cold-start users (no history) and items (no ratings)
- CSV files: `data/ratings.csv`, `data/users.csv`, `data/items.csv`

## Success Criteria
- **RMSE**: < 1.0 on test ratings (existing users)
- **Cold-start**: At least 4/5 reasonable recommendations for cold-start users
- **Precision@10**: > 0.60
- **Diversity**: Average pairwise dissimilarity > 0.3 within recommendation lists

## Starter Code
`starter_code.py` loads data and provides a basic SVD (Singular Value Decomposition) approach. You must extend with hybrid blending.

## Constraints
- Must handle both warm-start and cold-start scenarios
- Must combine collaborative filtering and content-based signals
- Must evaluate both prediction accuracy and recommendation quality
