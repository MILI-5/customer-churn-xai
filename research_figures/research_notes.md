# 2. Introduction

Machine learning systems are increasingly used in business-critical domains such as customer churn prediction, fraud detection, healthcare analytics, and financial risk assessment. Although these models achieve high predictive accuracy, many advanced algorithms operate as “black boxes,” making their decisions difficult for humans to interpret.

In business environments, explainability is essential because stakeholders need to understand why a model made a particular prediction before taking action. Regulatory frameworks such as the General Data Protection Regulation (GDPR) also emphasize the “right to explanation” for automated decisions, increasing the demand for interpretable artificial intelligence systems.

Traditional explainable AI (XAI) techniques such as SHAP and LIME provide mathematical feature importance explanations. However, these outputs are often difficult for non-technical business users to understand directly. Technical values and visualization plots alone may not effectively support decision-making processes.

This project proposes a hybrid Explainable AI and Generative AI pipeline for customer churn prediction. The system combines machine learning models, SHAP-based feature attribution, LIME local explanations, and AI-generated business insights to create human-readable explanations for churn predictions.

The proposed pipeline bridges the gap between technical model interpretability and practical business understanding. By transforming numerical explanations into natural-language recommendations, the system improves decision support for customer retention strategies.

# 3.Related work

Explainable Artificial Intelligence (XAI) has gained significant attention in recent years due to the widespread adoption of machine learning models in high-impact decision-making systems. While traditional machine learning algorithms provide strong predictive performance, their lack of interpretability limits their usability in domains where transparency and accountability are essential.

A foundational contribution in this area is SHAP (SHapley Additive exPlanations), introduced by Lundberg and Lee (2017). SHAP is based on cooperative game theory and assigns each feature an importance value for individual predictions. It provides both local and global interpretability, making it one of the most widely adopted post-hoc explanation techniques in machine learning systems.

Similarly, LIME (Local Interpretable Model-agnostic Explanations), proposed by Ribeiro et al. (2016), generates explanations by approximating complex models locally using interpretable surrogate models. LIME is model-agnostic and is particularly useful for explaining individual predictions, even when the underlying model is highly complex.

Beyond these techniques, recent research has explored the role of Explainable AI in business decision support systems. Adadi and Berrada (2018) emphasize that interpretability is essential for building trust in AI systems, particularly in domains such as finance, healthcare, and customer analytics. In churn prediction systems, explainability enhances stakeholder confidence and enables actionable decision-making based on model outputs.

In parallel, advancements in Large Language Models (LLMs) and Generative AI have introduced new possibilities for transforming machine learning explanations into natural language narratives. Instead of presenting raw feature importance values or technical visualizations, generative models can translate these outputs into human-readable insights and business recommendations. This improves accessibility for non-technical stakeholders and supports better decision-making.

Despite these advancements, most existing research focuses either on interpretability techniques (such as SHAP or LIME) or on natural language generation separately. There is limited work on integrating both explainability and generative reasoning into a unified pipeline for business applications. This research addresses that gap by combining machine learning models, SHAP, LIME, and Generative AI to create a complete explainable churn prediction system.


