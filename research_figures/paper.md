# Explainable Customer Churn Prediction Using Machine Learning, SHAP, LIME, and Generative AI

Author: Sanjana Waghmare  
Domain: Artificial Intelligence / Explainable AI  

## Abstract

Machine learning models are increasingly used in business decision-making systems such as customer churn prediction. However, many high-performing models operate as black boxes, making their predictions difficult to interpret for non-technical stakeholders. This research proposes an explainable artificial intelligence pipeline integrating Machine Learning, SHAP, LIME, and Generative AI for business-readable customer churn analysis. Multiple machine learning models including Logistic Regression, Random Forest, XGBoost, and Artificial Neural Networks were evaluated on the Telco Customer Churn dataset. Experimental results showed that the Artificial Neural Network achieved the highest accuracy of 79.60% and an AUC score of 0.835. SHAP and LIME were used to generate local and global explanations for model predictions, while a Generative AI layer translated technical outputs into human-understandable business insights and retention recommendations. The proposed system improves interpretability, transparency, and practical usability of predictive analytics in business environments.

## 1. Introduction

Customer churn prediction has become one of the most important applications of machine learning in modern business analytics. Telecommunication companies, banking institutions, subscription platforms, and online services use predictive models to identify customers who are likely to discontinue services. Early detection of churn enables organizations to reduce customer loss through personalized retention strategies.

Despite achieving high predictive accuracy, many advanced machine learning models operate as black-box systems. Business managers often receive predictions without understanding the reasoning behind them. This lack of transparency creates trust issues, especially in high-stakes decision-making environments.

Explainable Artificial Intelligence (XAI) addresses this challenge by providing interpretable insights into machine learning predictions. Techniques such as SHAP and LIME help identify the contribution of individual features toward model decisions. However, technical explainability outputs are often difficult for non-technical stakeholders to interpret.

This research proposes a novel explainable AI pipeline that combines machine learning, SHAP, LIME, and Generative AI to create business-readable customer churn explanations. The system not only predicts churn risk but also generates understandable business insights and retention recommendations for decision-makers.

## 2. Related Work

Explainable Artificial Intelligence (XAI) has become an important research area due to the increasing adoption of machine learning systems in critical decision-making applications. Traditional machine learning models often achieve high predictive performance but lack interpretability, making it difficult for users to understand the reasoning behind predictions.

Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a unified framework for interpreting machine learning predictions using concepts from cooperative game theory. SHAP provides both local and global interpretability and has become one of the most widely adopted explainability techniques in AI research.

Ribeiro et al. (2016) proposed LIME (Local Interpretable Model-Agnostic Explanations), which explains individual predictions by approximating complex models locally with interpretable surrogate models. LIME enables users to understand the factors influencing specific predictions regardless of the underlying model architecture.

Recent studies have explored the application of explainable AI in business analytics, particularly in customer behavior prediction and financial risk assessment. These systems improve transparency and increase stakeholder trust in AI-driven decisions.

Large Language Models (LLMs) and Generative AI systems are also increasingly being integrated into analytical workflows to convert technical outputs into human-readable explanations. However, limited research exists on combining SHAP, LIME, and Generative AI into a unified business-oriented explainability pipeline. This research attempts to bridge that gap.

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NeurIPS).

2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

3. Molnar, C. (2022). Interpretable Machine Learning. Lulu.com.

4. Adadi, A., & Berrada, M. (2018). Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI). IEEE Access.