# CampusCare AI — Big Data Student Support Message Triage

CampusCare AI is a deep learning course project for routing anonymous student support messages into the correct human-support queue.

The project solves a realistic university problem: student feedback/advising systems receive thousands of messages, especially near exams. A triage model can help staff prioritize messages while keeping a human in the loop.

> Educational prototype only. This project is **not** a diagnosis tool and must never replace counselors, advisors, emergency services, or human judgment.

## What changed in this big-data version

This version includes a large offline dataset inside the project, so it does not depend on internet download:

- `data/campuscare_large_dataset.csv` — **60,000 messages**
- `data/campuscare_challenge_test.csv` — **1,200 hard/ambiguous messages** for stronger error analysis
- Updated loader to use the big dataset automatically
- Faster TF-IDF + SGD Logistic Regression baseline
- TextCNN and BiLSTM deep models
- Optional DistilBERT Transformer script
- Updated report, rubric alignment, dataset card, experiment outputs, plots, and demo app

## Labels

| Label | Meaning | Routing action |
|---|---|---|
| `normal` | neutral, positive, or routine student message | normal support queue |
| `academic_stress` | study pressure, exam panic, grade/workload stress | academic advisor/tutor |
| `emotional_distress` | sadness, isolation, hopelessness, low mood | counseling/wellbeing support |
| `urgent_human_review` | safety concern or immediate risk wording | immediate trained-human review |

## Dataset distribution

| Label | Rows |
|---|---:|
| normal | 24,000 |
| academic_stress | 18,000 |
| emotional_distress | 12,000 |
| urgent_human_review | 6,000 |
| **Total** | **60,000** |

The dataset is custom synthetic, generated from realistic campus support-message templates plus augmentation/noise. It is designed for course experimentation, not clinical use.

## Project structure

```text
CampusCare_AI_BigData_Project/
├── app/streamlit_app.py
├── data/
│   ├── campuscare_large_dataset.csv
│   ├── campuscare_challenge_test.csv
│   ├── dataset_card.md
│   └── label_schema.json
├── models/
├── notebooks/01_campuscare_big_data_full_project.ipynb
├── outputs/
├── reports/
│   ├── project_report.md
│   ├── rubric_alignment.md
│   └── experiment_log_template.csv
└── src/
    ├── config.py
    ├── data_utils.py
    ├── generate_big_dataset.py
    ├── train_baseline.py
    ├── train_deep.py
    ├── train_transformer.py
    └── make_plots.py
```

## How to run

Install requirements:

```bash
pip install -r requirements.txt
```

Run the baseline:

```bash
python src/train_baseline.py
```

Run deep models:

```bash
python src/train_deep.py --models TextCNN,BiLSTM --epochs 2
```

For a faster CPU demo:

```bash
python src/train_deep.py --models TextCNN --epochs 2 --max-train-samples 12000
python src/train_deep.py --models BiLSTM --epochs 1 --max-train-samples 4000
```

Generate plots:

```bash
python src/make_plots.py
```

Run the demo app:

```bash
streamlit run app/streamlit_app.py
```

Optional Transformer experiment, if internet/model cache is available:

```bash
python src/train_transformer.py
```

## Included experiment results

These outputs are already generated in `outputs/`.

| Model | Split | Accuracy | Macro F1 |
|---|---|---:|---:|
| TF-IDF + SGD Logistic Regression | test | 1.000 | 1.000 |
| TF-IDF + SGD Logistic Regression | challenge | 0.865 | 0.864 |
| TextCNN | test | 1.000 | 1.000 |
| TextCNN | challenge | 0.744 | 0.723 |
| BiLSTM | test | 0.535 | 0.475 |
| BiLSTM | challenge | 0.381 | 0.339 |

Important interpretation: the normal test split comes from the same synthetic-generation distribution, so scores can be very high. The challenge split is more useful for error analysis because it contains ambiguous messages such as “I am safe, but…” or “this is academic, not crisis support.”

## Main files to open first

1. `notebooks/01_campuscare_big_data_full_project.ipynb`
2. `reports/project_report.md`
3. `reports/rubric_alignment.md`
4. `outputs/model_comparison.png`
5. `outputs/baseline_challenge_mistakes.csv`
6. `outputs/textcnn_challenge_mistakes.csv`

## Suggested team split

- Member 1: data curation, dataset card, preprocessing, EDA.
- Member 2: TF-IDF baseline and challenge-set analysis.
- Member 3: TextCNN model and CNN justification.
- Member 4: BiLSTM model and sequence-model discussion.
- Member 5: Transformer script, final report, demo app, presentation.

## Academic integrity note

Use this as a learning scaffold. Run it yourself, keep your own experiment log, update results if you retrain, and make sure every team member can explain their model and design choices.
