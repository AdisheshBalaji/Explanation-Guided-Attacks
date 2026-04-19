# Evaluating XAI Robustness via Adversaries

## Overview

This project studies the reliability of post-hoc explanation methods such as SHAP and LIME.

These methods are widely used to interpret machine learning models in high-stakes domains like finance, healthcare, and hiring. The central question is:

> Can we trust these explanations, or can they be manipulated?

---

## Problem Statement

Modern ML models are often black-box systems. To improve transparency, explanation methods assign importance scores to input features.

These explanations are used for:
- Fairness auditing  
- Debugging models  
- Regulatory compliance  

However, both SHAP and LIME rely on perturbing input data and observing model behavior.

### Core Issue

- Perturbations often lie outside the true data distribution (off-manifold)  
- Models are not trained on these inputs  
- Model behavior in these regions is unreliable  

**Conclusion:**  
Explanations depend on model behavior where the model is unconstrained.

---

## Key Idea

We investigate whether it is possible to manipulate explanations without changing model predictions.

Goal:
- Model remains accurate and fair  
- Explanation falsely indicates bias  

This leads to explanation-guided adversarial attacks.

---

## Attacks and Techniques

### 1. Scaffolding Attack

Model is split into:
- A fair model for real inputs  
- A biased model for perturbed inputs  

A detector decides which model to use.

Result:
- Predictions remain correct  
- Explanations become misleading  

---

### 2. Gradient-Based Manipulation

Modify training objective to increase sensitivity w.r.t. a sensitive feature.

Effect:
- High local gradients  
- Explanation methods interpret this as importance  

Key insight:
- Local sensitivity ≠ global dependence  

---

### 3. Optimization-Based Attacks (Direction)

Formulate attack as:

- Keep prediction unchanged  
- Maximize importance of sensitive feature  

Provides a principled attack framework.

---

### 4. Data Manifold Perspective

Real data lies on a low-dimensional manifold with correlated features.

Problem:
- LIME/SHAP sample in full space  
- Ignore correlations  

This mismatch is a major vulnerability.

---

## Relevant Papers

### Core Papers
- Fooling LIME and SHAP  
- Interpretation of Neural Networks is Fragile  
- Explanations can be manipulated and geometry is to blame  
- You Shouldn’t Trust Me  
- Why Should I Trust You? (LIME)  
- A Unified Approach to Interpreting Model Predictions (SHAP)  

### Attack Frameworks
- The Inverted Retraining Framework  
- Reverse Scaffolding  

### Evaluation
- OpenXAI  

### Additional
- On the Robustness of Removal-Based Feature Attributions  
- Evolutionary Algorithm for Black-Box Attacks on Explainability Methods  

---

## Professor’s Directions

### 1. SHAP and LIME Specific Techniques
- Understand internal mechanisms  
- Perturbation strategies  
- Sampling assumptions  

---

### 2. Game-Theoretic Alternatives

Explore alternatives to Shapley values:
- Banzhaf value  
- Nucleolus  

Potentially more stable in fairness settings.

---

### 3. Attack Robust Methods

Go beyond SHAP/LIME:
- Integrated Gradients  
- Removal-based methods  
- Gradient-based explainers  

---

### 4. Counter Scaffolding Attacks

- Detect manipulation on perturbed inputs  
- Identify inconsistencies between real and perturbed behavior  

---

### 5. Alternatives to Shapley-Based Methods

Find explanation frameworks less dependent on perturbation assumptions.

---

## Defense Ideas

### 1. Integrated Gradients
- More stable than perturbation-based methods  

---

### 2. Bayesian Explanations
- BayLIME / BaySHAP  
- Provide uncertainty estimates  

---

### 3. Stability Checks
- Add noise  
- Measure variance in explanations  

---

### 4. Manifold-Aware Methods (Direction)
- Restrict perturbations to realistic data  
- Avoid OOD sampling  

---

## Evaluation Strategy

Goal:
Show mismatch between predictions and explanations.

Check:
- Prediction unchanged  
- Feature importance changes significantly  
- Sensitive feature gains importance  

Metrics:
- Rank change  
- Attribution magnitude change  
- Attack success rate  

---

## Expected Outcome

- Model remains accurate and fair  
- Explanation shows false bias  

Conclusion:
Explanations can be manipulated independently of model behavior.

---

## Implications

- Explanations may not be reliable for fairness auditing  
- Models can be falsely flagged as biased  
- Risks in regulatory and real-world applications  

---

## Possible Extensions

- Compare multiple explanation methods  
- Study on-manifold vs off-manifold perturbations  
- Analyze stability vs interpretability  
- Develop robust explanation methods  

---

## Summary

This project exposes a key weakness in explainability methods:

They can be adversarially manipulated without affecting model predictions.

The goal is to evaluate which methods are reliable and explore more robust alternatives.