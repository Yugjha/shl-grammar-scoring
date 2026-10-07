# SHL Grammar Scoring Engine

## Project Overview
Grammar Scoring Engine for spoken audio samples using Machine Learning.

## Problem Statement
Develop a model that predicts grammar scores (0-5) from audio files (45-60 seconds each).

## Dataset
- **Training samples:** 769 audio files with labels (1-5)
- **Test samples:** 216 audio files
- **Audio format:** .wav files

## Solution Approach

### 1. Feature Extraction
- **MFCC (Mel-Frequency Cepstral Coefficients):** 13 features
  - Statistics: Mean, Std Dev, Max, Min (4 per MFCC)
  - Total MFCC features: 52
- **Spectral Features:** 3 features
  - Spectral Centroid
  - Spectral Rolloff
  - Zero Crossing Rate
- **Total Features:** 55

### 2. Preprocessing
- StandardScaler normalization (mean=0, std=1)

### 3. Model Training
Tested 3 models:
| Model | RMSE | R² | Correlation |
|-------|------|-----|-------------|
| RandomForest | 0.2981 | 0.9420 | 0.9780 |
| **GradientBoosting** | **0.1666** | **0.9819** | **0.9927** |
| Ridge | 0.8032 | 0.5792 | 0.7611 |

**Selected:** GradientBoostingRegressor (Best performance)

## Results

### Training Metrics
- **RMSE:** 0.1666
- **R² Score:** 0.9819 (98.19% variance explained)
- **Pearson Correlation:** 0.9927 (nearly perfect)

### Test Predictions
- **Total predictions:** 216
- **Score range:** [2.18, 5.00]
- **Mean score:** 3.24

### Kaggle Score
- **Leaderboard Score:** 0.7822

## Files
- `notebook.ipynb` - Complete code with analysis
- `submission.csv` - Test predictions
- `README.md` - This file

## Technologies Used
- Python 3
- Librosa (audio processing)
- Scikit-learn (ML models)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualizations)

## How to Run
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook to train model and generate predictions

## Key Findings
- GradientBoosting significantly outperforms other models
- MFCC + Spectral features capture grammar patterns effectively
- Model achieves near-perfect correlation on training data
- Low RMSE indicates high prediction accuracy

## Author
Bhartendu Jha

## License
MIT
