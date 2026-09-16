# Notebooks Directory (Supplementary Only)

Per academic and software engineering best practices, this project is architected as a **CLI-first, fully reproducible Python package**.

All primary workflows are executed directly via terminal commands:
- Data preparation: `python -m src.main prepare-data`
- Training: `python -m src.main train` (or `python -m src.main train --quick`)
- Evaluation: `python -m src.main evaluate`
- Inference: `python -m src.main predict --image <path>`
- System validation: `python -m src.main validate`

Notebooks are entirely optional and supplementary. No step in this project requires launching or executing Jupyter notebooks.
