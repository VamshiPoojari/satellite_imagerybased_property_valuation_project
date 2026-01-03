# Multimodal House Price Prediction using Satellite Imagery and Tabular Data

## Overview
This project builds a **multimodal regression pipeline** to predict residential property prices by combining traditional tabular housing features with **satellite imagery–based visual context**.  
While standard real-estate models rely only on structured attributes (e.g., bedrooms, square footage), this project incorporates **neighborhood-level visual cues** such as green cover, road density, and urban layout using Convolutional Neural Networks (CNNs).

The system integrates:
- Tabular data modeling (Linear Regression, XGBoost)
- Satellite image acquisition using geographic coordinates
- CNN-based visual feature extraction
- Multimodal late-fusion modeling
- Visual explainability using Grad-CAM

---

## Dataset
- **Base Dataset:** Kaggle House Sales dataset  
- **Target Variable:** `price`
- **Key Tabular Features:**  
  `bedrooms, bathrooms, sqft_living, sqft_lot, floors, condition, grade, lat, long`
- **Visual Data:**  
  Satellite images fetched programmatically using latitude and longitude via **Mapbox Static Images API**

---


## Methodology

### 1. Tabular Modeling
- Linear Regression used as a baseline
- XGBoost used for strong non-linear tabular performance

### 2. Satellite Image Acquisition
- Satellite images downloaded using Mapbox Static Images API
- One image per property based on latitude and longitude
- Images resized to 224×224 for CNN compatibility

### 3. Image Feature Engineering
- Pretrained **ResNet18** (ImageNet) used as feature extractor
- Final classification layer removed
- Each image converted into a 512-dimensional embedding
- PCA applied to reduce image features to 50 dimensions (noise control)

### 4. Multimodal Fusion
- Late fusion approach
- Tabular features concatenated with image embeddings
- XGBoost used as final regression model

### 5. Explainability
- Grad-CAM applied to CNN layers
- Visual heatmaps highlight influential regions such as:
  - Green areas
  - Roads
  - Built-up regions

---

## Results

| Model | RMSE | R² |
|-----|-----|----|
| Linear Regression (Tabular) | ~219k | 0.62 |
| XGBoost (Tabular) | ~139k | 0.845 |
| Multimodal (Raw Image Embeddings) | ~156k | 0.81 |
| Multimodal (PCA-Controlled) | ~143k | 0.838 |

While the multimodal model did not significantly outperform the strongest tabular-only model in RMSE, it provided **valuable interpretability and neighborhood-level insights** not available in structured data alone.

---

## Explainability
Grad-CAM visualizations reveal that the CNN focuses on:
- Green cover (trees, parks)
- Road networks and accessibility
- Dense built-up structures

These regions align with known real-estate valuation drivers, validating the usefulness of visual context.
