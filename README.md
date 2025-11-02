This repository contains the findings for the "Adobe Behavior Simulation Challenge." This project tackles two critical challenges in social media analytics:

##Task 1: Behavior Simulation: Predicting a tweet's engagement (like count) by solving the "engagement heterogeneity problem" (the vast difference between viral and non-viral content).
##Task 2: Content Simulation: Generating authentic, brand-specific tweet text based on a given set of metadata, including a media URL.

#Task 1: Behavior Simulation (Engagement Prediction):

This task predicts the log-transformed number of likes a tweet will receive. Traditional regression models struggle with this due to the extremely skewed distribution of likes.

##The Challenge: Engagement Heterogeneity
A single model cannot effectively learn patterns across the entire engagement spectrum, from 10 likes to 100,000 likes. Our solution addresses this by first classifying the "regime" of engagement and then performing a specialized regression.

##Architecture: Two-Stage Hybrid Model
We developed a novel, two-stage hybrid architecture built on LightGBM that significantly outperforms standard regression.

##Stage 1: Engagement Bin Classifier
A LightGBM Multiclass Classifier categorizes each tweet into one of five engagement bins (Very Low, Low, Medium, High, Very High) based on quantile-based splits.

##Stage 2: Bin-Augmented Regressor
A LightGBM Regressor predicts the precise log-like count.

##Key Innovation: The regressor is fed both the original features and the 5-dimensional probability distribution from the Stage 1 classifier. This "soft-gating" mechanism allows the regressor to understand the uncertainty of the initial bin prediction and learn bin-specific relationships.

##Features:
The model uses a multi-modal feature set:
Content Embedding (768-dim): Represents the semantics, tone, and topic of the tweet text.
User-Company Embedding (90-dim): A PCA-reduced embedding representing brand identity and audience.
Temporal Features (6-dim): Cyclical sine/cosine encodings for hour, day of the week, and month.
Content Features (1-dim): word_count.

##Key Results (Task 1):
Standard Split (Known Brands):
R²: 0.8724 (The model explains 87.2% of the variance in engagement).
Log RMSE: 0.9268
Log MAE: 0.5954
Unseen Brands Split (Generalization):
R²: 0.0428

Key Insight: The dramatic performance drop proves that brand identity is a dominant factor in engagement prediction. The model learned brand-specific patterns, highlighting the difficulty of generalizing to new clients without transfer learning.

#Task 2: Content Simulation (Tweet Generation)

This task generates authentic tweet text given metadata like company, username, timestamp, and a media_url.

##The Challenge: The Context Gap
A generative model cannot interpret a media_url string. It doesn't understand the visual content of the image, leading to generic or irrelevant text. Our solution "bridges this context gap" by translating the image into text.

##Architecture: Context-Enrichment Pipeline
We developed a multi-stage pipeline to enrich the input data before fine-tuning a generative model.

##Phase 1: Contextual Enrichment (Image Captioning)
A visual-language model (Salesforce/blip-image-captioning-base) is used to "see" the image at each media_url.
It generates a descriptive text caption (e.g., "a red sports car on a winding road"), effectively translating the visual context into a textual feature.

##Phase 2: Prompt Engineering
We construct detailed, structured prompts that include all available metadata:
Date:
Company:
Username:
Image Caption: (The newly generated text)

##Phase 3: Model Specialization (Fine-Tuning)
An openai/gpt-oss-20b transformer model is fine-tuned on the structured (prompt, completion) pairs.
Prompt Masking is used to ensure the model only learns to predict the completion (the original tweet) based on the prompt context.

##Key Results (Task 2):
The fine-tuned model produced high-quality, context-aware text. The strong overlap with the ground-truth tweets validates the effectiveness of the context enrichment pipeline.
ROUGE-L (F1-Score Avg): 0.9788
ROUGE-2 (F1-Score Avg): 0.8655
BLEU-4 (Cumulative): 0.8341

These scores demonstrate that the model successfully learned to align generated content with the brand identity, timestamp, and, most importantly, the visual content of the accompanying image.

