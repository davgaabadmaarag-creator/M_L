# Өсвөр үеийнхний сэтгэцийн эрүүл мэндийн ML төсөл

Энэ төсөл нь өсвөр үеийнхний сошиал медиа хэрэглээ болон амьдралын хэв маягийн үзүүлэлтүүдээр сэтгэцийн эрүүл мэндийн эрсдэлийг `Low`, `Medium`, `High` гэж ангилах машин сургалтын жишээ төсөл юм.

## Ажиллуулах

Төслийн үндсэн хавтаснаас:

```powershell
python ml\stat.py
```

Эсвэл `ml` хавтас руу орж:

```powershell
python stat.py
```

## Нэг хүний мэдээлэл оруулж шинжилгээ хийх

```powershell
python ml\stat.py --interactive
```

## Combined screen addiction app

The project also includes the merged self-test app from
`screen_addiction_student_test`.

From the project root:

```powershell
python ml\screen_addiction_app.py --cli
```

From Git Bash:

```bash
python ml/screen_addiction_app.py --cli
```

To open the GUI version on Windows:

```powershell
python ml\screen_addiction_app.py
```

The self-test app saves its latest output to:

- `ml/results/screen_addiction_prediction.txt`
- `ml/results/screen_addiction_user_input.csv`

Энэ горим нь нас, сошиал медиа ашигласан цаг, унтах цаг, унтахын өмнөх дэлгэцийн цаг, хөдөлгөөн, платформ, нийгмийн харилцааны түвшин зэрэг мэдээллийг асуугаад тухайн хүний эрсдэлийн ангиллыг таамаглана.

## Үр дүн

Бүх үр дүн `ml/results/` хавтаст хадгалагдана. Гол файлууд:

- `model_results.txt` - загваруудын дэлгэрэнгүй metric
- `accuracy_comparison.csv` - Accuracy, Balanced Accuracy, Macro F1 харьцуулалт
- `practical_insights.txt` - стресс өндөр бүлгийн практик тайлбар
- `platform_stress_summary.csv` - платформ бүрийн стрессийн харьцуулалт
- `social_media_by_stress_level.csv/.png` - стресс ба сошиал хэрэглээний хамаарал
- `user_prediction.txt` - interactive горимын таамаглал

## Анхаарах зүйл

Загвар сонгохдоо энгийн accuracy биш `Macro F1`-ийг гол metric болгосон. Учир нь dataset дээр `Medium` ангилал олон байгаа тул model бүх мөрийг `Medium` гэж таамаглахад accuracy өндөр мэт харагдаж болно.

Энэ төсөл нь сургалтын зориулалттай статистик болон машин сургалтын шинжилгээ юм.
