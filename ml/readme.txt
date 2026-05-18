Teen Mental Health ML Project

Goal:
Predict teen mental health risk level from social media usage and lifestyle features.

Target:
mental_health_risk_score = mean(stress_level, anxiety_level, addiction_level)

Risk labels:
Low    = 1.00 - 3.99
Medium = 4.00 - 6.99
High   = 7.00 - 10.00

Run the full analysis:
python stat.py

Run the interactive single-profile analysis:
python stat.py --interactive

Main outputs in results/:
- model_results.txt
- accuracy_comparison.csv / accuracy_comparison.png
- practical_insights.txt
- risk_level_profile.csv
- stress_group_profile.csv
- platform_stress_summary.csv
- social_media_by_stress_level.csv / .png
- decision_tree_feature_importance.csv / .png
- user_prediction.txt, only after --interactive

Important:
This project is for educational statistical and machine learning analysis.
It must not be used as a clinical diagnosis.
