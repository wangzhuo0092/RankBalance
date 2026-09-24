# Datasets

The release includes the CSV files referenced by
`configs/paper_datasets.json`.

| Dataset ID | Input file |
|---|---|
| `ihq_all` | `projects/clic2024/data/2AFC_google_elo.csv` |
| `chatbot_arena_33k` | `projects/chatbot_arena_33k/data/preferences.csv` |
| `vision_arena` | `projects/vision_arena/data/preferences.csv` |
| `search_arena` | `projects/search_arena/data/preferences.csv` |
| `computer_agent_arena` | `projects/computer_agent_arena/data/preferences.csv` |
| `mt_bench` | `projects/mtbench/data/human_judgments.csv` |
| `humaine` | `projects/prolific/data/feedback_comparisons.csv` |
| `multipref_all` | `projects/multipref/data/multipref_all.csv` |
| `llm_judge_holdout` | `projects/llm_judge_holdout/data/preferences.csv` |
| `hific` | `projects/hific/data/userstudy_google_elo.csv` |
| `conha` | `projects/conha/data/answers.csv` |
| `wd` | `projects/WD/data/answers.csv` |

The loaders normalize model names, binary outcomes, ties, and rater IDs before
fitting. Golden/control records are excluded where the source schema marks
them. No generated split manifests or fitted outputs are included.

These datasets originate from third-party projects and remain subject to their
respective upstream terms. Dataset citations and provenance are provided in the
associated paper.
