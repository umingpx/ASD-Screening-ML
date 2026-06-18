# Clinical Framework: Into the Spectrum

This document outlines the diagnostic mapping between the behavioral features used in this model and the established medical criteria.

## 1. Primary Framework: Q-CHAT-10
The model utilizes the Quantitative Checklist for Autism in Toddlers (Q-CHAT-10), a 10-item screening tool designed by Dr. Faisal Thabtah for toddlers aged 12-36 months.

## 2. DSM-5 Alignment
Every feature in the 'Into the Spectrum' architecture is mapped to the Diagnostic and Statistical Manual of Mental Disorders (DSM-5) criteria for Autism Spectrum Disorder:

### Criterion A: Social Communication and Interaction
*Deficits in social-emotional reciprocity, nonverbal communicative behaviors, and developing relationships.*
- **A1 [Social Attention]:** Response to name.
- **A2 [Eye Contact]:** Quality and frequency of gaze.
- **A3 [Requesting]:** Use of pointing to indicate needs.
- **A4 [Joint Attention]:** Use of pointing to share interest.
- **A5 [Pretend Play]:** Engagement in imaginative play.
- **A6 [Gaze Following]:** Following the gaze of others.
- **A9 [Gestures]:** Use of simple gestures (e.g., waving).

### Criterion B: Restricted, Repetitive Patterns of Behavior
*Stereotyped motor movements, insistence on sameness, or highly fixated interests.*
- **A7 [Social-Emotional Reciprocity]:** sign of comforting others (Empathy).
- **A8 [Language Development]:** Nature of first words (Typical vs. Atypical).
- **A10 [Repetitive/Sensory]:** Presence of repetitive movements or blank staring.

## 3. Implementation Note
The Random Forest model assigns mathematical weights to these criteria based on the training dataset (n=1,054). As established in our results analysis, features A9 and A7 (Criterion A) were identified as the highest-weight predictors in this specific cohort.
