**Project Overview**

This project develops a county-level forecasting framework for wholesale white maize prices by integrating heterogeneous agricultural market datasets from KAMIS and AgriBORA. The objective is to learn a stable mapping between the two price systems and generate short-horizon forecasts under temporal and spatial constraints.
The pipeline combines time-series aggregation, statistical comparison, spatial feature construction, and regularized regression modeling within a unified workflow.

________________________________________________________________________________________________________________________________________________________________________________________________________
**Data Preparation and Alignment**
Both datasets are filtered to retain observations corresponding to White Maize and restricted to selected Kenyan counties. Price variables are standardized and timestamps are converted into a common weekly index.
Weekly aggregation produces aligned county–time panels, enabling direct comparison between datasets. Data coverage and overlap are analyzed to assess temporal consistency and sampling bias across regions.
Exploratory analysis includes visualization of coverage density and county-wise price trajectories, alongside statistical agreement metrics such as correlation, bias, and mean absolute deviation.

________________________________________________________________________________________________________________________________________________________________________________________________________
**Spatial and Temporal Feature Construction**
County centroid coordinates are incorporated to encode geographic structure. Pairwise distances are computed using the Haversine formulation, allowing identification of spatial neighbors and regional proximity relationships.
A continuous weekly panel is constructed for each county. Missing observations are imputed via forward–backward propagation, followed by smoothing to reduce high-frequency volatility.
Temporal dependence is modeled through lagged features derived from historical price dynamics:
x_t,; x_{t-1},; x_{t-2},; x_{t-3}

forming a structured autoregressive representation.

________________________________________________________________________________________________________________________________________________________________________________________________________
**Modeling Framework**
Prediction is performed using an Elastic Net regression pipeline, chosen to balance sparsity and coefficient stability under multicollinearity.
The modeling pipeline includes:
•	numerical imputation and standardization,
•	categorical encoding of counties,
•	joint L1–L2 regularization.
The model estimates the relationship between smoothed KAMIS prices and corresponding AgriBORA prices within a supervised learning framework.
Model performance is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

________________________________________________________________________________________________________________________________________________________________________________________________________
**Forecasting Strategy**
Future prices are generated via recursive multi-step forecasting. Predicted values are iteratively fed back as lag inputs, enabling projection beyond observed data while preserving temporal structure.
Forecasts are produced for multiple counties simultaneously and formatted into a standardized submission schema indexed by ISO calendar weeks.

________________________________________________________________________________________________________________________________________________________________________________________________________
**Contributions**
•	Integration of multi-source agricultural price systems
•	Weekly panel time-series construction with temporal alignment
•	Spatially informed preprocessing using geographic distances
•	Regularized regression for stable economic signal extraction
•	Recursive forecasting pipeline for short-term market prediction

________________________________________________________________________________________________________________________________________________________________________________________________________
