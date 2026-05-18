Өсвөр үеийнхний сэтгэцийн эрүүл мэндийн ML төсөл

Зорилго:
Өсвөр үеийнхний сошиал медиа хэрэглээ болон амьдралын хэв маягийн үзүүлэлтүүдээр
сэтгэцийн эрүүл мэндийн эрсдэлийг Low / Medium / High гэж ангилах.

Target үүсгэх арга:
mental_health_risk_score = mean(stress_level, anxiety_level, addiction_level)

Эрсдэлийн ангилал:
Low    = 1.00 - 3.99  буюу бага эрсдэл
Medium = 4.00 - 6.99  буюу дунд эрсдэл
High   = 7.00 - 10.00 буюу өндөр эрсдэл

Бүх шинжилгээг ажиллуулах:
python stat.py

Төслийн үндсэн хавтаснаас ажиллуулах:
python ml\stat.py

Нэг хүний мэдээлэл оруулж таамаглал хийх:
python stat.py --interactive

results/ хавтаст гарах гол файлууд:
- model_results.txt
- accuracy_comparison.csv / accuracy_comparison.png
- model_metric_comparison.csv
- model_macro_f1_comparison.png
- practical_insights.txt
- risk_level_profile.csv
- stress_group_profile.csv
- platform_stress_summary.csv
- social_media_by_stress_level.csv / .png
- decision_tree_feature_importance.csv / .png
- user_prediction.txt, зөвхөн --interactive ажиллуулсны дараа

Загвар сонгох логик:
Скрипт нь Dummy Baseline-тай харьцуулж, baseline биш хамгийн сайн загварыг Macro F1 metric-ээр сонгоно.
Macro F1 ашиглаж байгаа шалтгаан нь нэг ангилал хэт олон үед энгийн accuracy буруу ойлголт өгч болдог.

Анхаарах зүйл:
Энэ төсөл нь сургалтын зориулалттай өгөгдлийн шинжилгээ, машин сургалтын жишээ юм.
Эмнэлзүйн онош тавих хэрэгсэл биш.
