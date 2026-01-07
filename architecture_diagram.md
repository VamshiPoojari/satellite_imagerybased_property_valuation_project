# Multimodal House Price Prediction – Architecture Diagram

## High-Level System Architecture

The system follows a **late-fusion multimodal architecture** that combines
structured tabular features with visual features extracted from satellite imagery.

---

## Architecture Overview
        ┌──────────────────────────────┐
        │     Tabular Housing Data     │
        │ (bedrooms, sqft, lat, long) │
        └───────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │        Preprocessing         │
        │ (Cleaning, EDA, Feature Sel) │
        └───────────────┬──────────────┘
                        │
                        ▼
                Tabular Feature Vector
                        │
                        │
                        │
        ┌──────────────────────────────┐
        │        Satellite Images      │
        │    (Lat/Long → Mapbox API)   │
        └───────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │     CNN Feature Extractor    │
        │   (ResNet18 – Pretrained)   │
        └───────────────┬──────────────┘
                        │
                        ▼
            Image Embeddings (512-d)
                        │
                        ▼
        PCA Dimensionality Reduction
                (512 → 50 dimensions)
                        │
                        ▼
                Image Feature Vector
                        │
                        └──────────────┐
                                    ▼
        ┌──────────────────────────────┐
        │       Multimodal Fusion      │
        │ (Concatenation of Tabular   │
        │        + Image Features)    │
        └───────────────┬──────────────┘
                        ▼
        ┌──────────────────────────────┐
        │      Regression Model        │
        │          (XGBoost)          │
        └───────────────┬──────────────┘
                        ▼
                Predicted House Price

## Component Description

### 1. Tabular Data Pipeline
- Input: Structured housing attributes
- Processing:
  - Missing value checks
  - Feature selection
- Models:
  - Linear Regression (baseline)
  - XGBoost (strong tabular baseline)


### 2. Satellite Image Pipeline
- Input: Latitude and Longitude
- Image Source: Mapbox Static Images API
- Processing:
  - Image resizing (224×224)
  - Normalization
- Feature Extraction:
  - Pretrained ResNet18 (ImageNet)
  - Final classification layer removed
  - Produces 512-dimensional embeddings
- Noise Control:
  - PCA applied to reduce embeddings to 50 dimensions

---

### 3. Multimodal Fusion
- Strategy: **Late Fusion**
- Tabular and image features concatenated
- Combined feature vector fed into XGBoost regressor

---

### 4. Explainability Module
- Grad-CAM applied to CNN layers
- Highlights influential regions in satellite imagery
- Provides visual justification for model decisions

---

## Design Rationale

- Late fusion allows independent optimization of tabular and visual pipelines
- PCA reduces image noise and prevents feature dominance
- Grad-CAM ensures transparency and interpretability

---

## Summary

This architecture enables the model to learn from both **structured property attributes**
and **unstructured visual context**, providing a robust and explainable framework for
real-estate price prediction.
