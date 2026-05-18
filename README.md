# Teen Mental Health ML Project

This project analyzes teen mental health risk using social media usage and lifestyle features.

## Run

```powershell
python ml\stat.py
```

You can also run it from inside the `ml` folder with `python stat.py`.

## Interactive prediction

```powershell
python ml\stat.py --interactive
```

The interactive mode asks for one person's age, social media usage, sleep, screen time, activity, platform, and social interaction level. It predicts Low/Medium/High risk and compares the entered profile with dataset averages and high-stress group averages.

## Outputs

Results are saved in `ml/results/`, including model metrics, practical insight summaries, high-stress platform analysis, lifestyle charts, and feature importance.

The model comparison includes a dummy baseline and selects the best non-baseline model by Macro F1 instead of plain accuracy. This is important because the dataset has more `Medium` labels than `Low` or `High`, so accuracy alone can be misleading.

This is an educational machine learning project, not a medical diagnosis tool.
