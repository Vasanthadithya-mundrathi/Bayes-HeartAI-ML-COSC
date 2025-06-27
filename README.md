# Bayes to the Future: Heart Disease Prediction with Bayesian Networks

## Challenge Description
Build a Bayesian Network to predict heart disease risk using a dataset of simulated patient records. Clean data, apply min-max normalization, build network with predefined structure, train using Maximum Likelihood Estimation, and answer diagnostic questions.

## Implementation Overview

### Dataset Processing
- **Records**: 303 patient records (302 after duplicate removal)
- **Features**: 13 input features + 1 target (heart disease)
- **Heart Disease Rate**: 54.30%

### Data Preprocessing
✅ **Data Cleaning**: Removed 1 duplicate record  
✅ **Missing Value Handling**: No missing values found  
✅ **Min-Max Normalization**: Applied to continuous features (age, trestbps, chol, thalach, oldpeak)  
✅ **Feature Discretization**: Converted continuous variables to categorical bins for Bayesian Network compatibility  

### Bayesian Network Structure
Implemented the required structure as specified in the challenge:

```
age → fbs → target
age → chol → target  
age → thalach → target
```

**Network Components:**
- **Nodes**: 5 (age_group, fbs, chol_category, hr_category, target)
- **Edges**: 6 directed edges representing causal relationships
- **Estimation**: Maximum Likelihood Estimation from training data

### Key Features

#### 1. Data Preprocessing Pipeline
- Duplicate removal and missing value handling
- Min-max normalization for all continuous features
- Feature discretization into meaningful categories:
  - Age: young, middle-aged, senior, elderly
  - Blood Pressure: normal, elevated, high, very_high
  - Cholesterol: normal, borderline, high
  - Heart Rate: low, normal, high
  - ST Depression: none, mild, severe

#### 2. Bayesian Network Implementation
- Custom implementation with conditional probability tables
- Maximum Likelihood Estimation for parameter learning
- Probabilistic inference for predictions

#### 3. Model Predictions
The model generates probabilistic predictions with confidence scores:
- Test Case 1 (High Risk): 29.9% probability of heart disease
- Test Case 2 (Low Risk): 7.2% probability of heart disease  
- Test Case 3 (Medium Risk): 29.3% probability of heart disease

#### 4. Health Insights
- **Age Factor**: Diabetes risk increases from 4.8% (young) to 18.2% (elderly)
- **High-Risk Profile**: High fasting blood sugar + high cholesterol + low heart rate = 80% disease risk
- **Risk Factors**: Clear probabilistic relationships between age, metabolic factors, and heart disease

### Technical Implementation

#### Files Structure
```
bayesian_heart_disease/
├── heart_disease.csv                    # Original dataset
├── heart_disease_predictor_simple.py    # Main implementation
├── processed_heart_disease_data.csv     # Cleaned dataset
├── heart_disease_eda.png               # Exploratory analysis plots
├── bayesian_network_structure.png      # Network visualization
├── Heart_Disease_Prediction_Report.md  # Detailed report
└── README.md                           # This file
```


#### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

#### Running the Code(setup)
```bash
cd bayesian_heart_disease/
python heart_disease_predictor_simple.py
or
python heart_disease_predictor.py
```

### Challenge Requirements Fulfilled

✅ **GitHub Repository**: Complete codebase with cleaned dataset  
✅ **Data Cleaning**: Removed duplicates, handled missing values  
✅ **Min-Max Normalization**: Applied to all continuous features  
✅ **Bayesian Network Structure**: Implemented predefined structure (age → fbs → target, chol, thalach)  
✅ **Maximum Likelihood Estimation**: Used for network parameter learning  
✅ **Inference Results**: Probabilistic predictions with health insights  
✅ **Visualizations**: Network structure and data analysis plots  
✅ **README with Setup Instructions**: Complete documentation  
✅ **Sample Outputs**: Demonstrates model predictions and insights  

### Model Capabilities

#### 1. Risk Assessment
- Calculates individual heart disease probability
- Provides confidence scores for predictions
- Identifies high-risk patient profiles

#### 2. Medical Insights
- Quantifies relationships between risk factors
- Explains probabilistic dependencies
- Generates actionable health recommendations

#### 3. Clinical Applications
- **Screening**: Early identification of at-risk patients
- **Decision Support**: Assists healthcare providers
- **Patient Education**: Explains risk factors clearly
- **Research**: Studies cardiovascular risk interactions

### Methodology

#### Network Design
The Bayesian Network captures key medical relationships:
- **Age** influences metabolic factors (fasting blood sugar, cholesterol, heart rate)
- **Metabolic factors** directly impact heart disease risk
- **Probabilistic reasoning** handles uncertainty in medical diagnosis

#### Parameter Learning
- Uses Maximum Likelihood Estimation
- Calculates conditional probability tables from training data
- Handles sparse data with reasonable default values

#### Inference Engine
- Variable elimination for probabilistic queries
- Handles complex evidence combinations
- Provides interpretable probability outputs

### Results and Validation

#### Model Performance
- Successfully processes 302 patient records
- Generates consistent probabilistic predictions
- Identifies meaningful risk patterns in data

#### Health Insights Generated
1. **Age Progression**: Clear increase in diabetes risk with age
2. **Risk Combinations**: High-risk profiles identified (80% disease probability)
3. **Protective Factors**: Low-risk combinations highlighted

#### Visualization Outputs
- **EDA Plots**: Comprehensive data exploration (6 subplot analysis)
- **Network Structure**: Clear visualization of causal relationships
- **Results Dashboard**: Organized presentation of findings

This implementation successfully demonstrates probabilistic modeling for medical diagnosis, providing both technical rigor and practical health insights.
