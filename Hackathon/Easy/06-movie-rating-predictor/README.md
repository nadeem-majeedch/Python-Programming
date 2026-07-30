# Problem 06 – Movie Rating Predictor

**Domain**: Regression  
**Goal**: Predict IMDb ratings based on movie attributes like genre, runtime, budget, and release year.

## Objectives
- Build a regression model using one-hot encoded genre.
- Apply Ridge regression to reduce overfitting.
- Create interaction features and evaluate their impact.
- Analyze residuals and feature importance.

## Success Criteria
- Achieve RMSE < 1.0 on the test set.
- R² score > 0.40 (ratings are noisy!).
- Identify which genre has the highest impact on rating.

## Dataset: `data/movie_ratings.csv`
~1000 rows with columns:
- MovieID, Genre, Runtime (min), Budget (M$), ReleaseYear
- IMDbRating (target)
