# Pediatric ASD Screening through Machine Learning
*Research project awarded Merit at the Young Scientist Competition (YSC) Thailand.*

## Introduction
The goal of this research was to develop a high-sensitivity screening tool for identifying Autism Spectrum Disorder (ASD) traits in toddlers aged 12 to 36 months. Originally conceived as a rule-based system in C++, this current iteration utilizes a Random Forest architecture to analyze behavioral patterns based on the Q-CHAT-10 clinical framework.

## Clinical Rationale and Dataset
The model is trained on a dataset of 1,054 cases, where inputs are mapped directly to the diagnostic criteria outlined in the DSM-5:
*   Social Communication (Criterion A): Assessed through features A1-A6 and A9 (e.g., joint attention and gesturing).
*   Restricted/Repetitive Behaviors (Criterion B): Assessed through features A7, A8, and A10.

To ensure the integrity of the machine learning process, a 'leakage fix' was implemented. The aggregate Q-CHAT score was removed from the training features, forcing the model to identify correlations between individual behaviors and the final classification rather than relying on a pre-calculated sum

## Results Analysis
During testing on a 20% hold-out sample, the Random Forest classifier achieved a recall (sensitivity) of 0.97 and an overall accuracy of 0.96. 

In a medical screening context, the 0.97 recall is the primary metric of success, as it suggests a 97% reliability rate in identifying children who may require further professional evaluation. Feature importance analysis indicated that A9 (gestural communication) and A7 (social-emotional reciprocity) were the most influential predictors in the model's decision-making process.

## Implementation Details
This repository contains the following components for peer review and replication:
*   `app.py`: A Streamlit-based web interface for clinical interaction.
*   `asd_model.pkl`: The serialized model weights for the Random Forest classifier.
*   `Processed_Dataset.csv`: The encoded toddler dataset used for training and validation.
*   `feature_importance.png`: A visual distribution of behavioral feature weights.
  
---
*Disclaimer: This is a screening tool designed for risk stratification and educational planning. It is not a substitute for professional clinical diagnosis.*
