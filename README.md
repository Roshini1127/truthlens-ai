# TruthLens AI v5.1 — Dark Violet Edition

## Features
- Purple/lavender gradient interface
- Google News RSS search and source comparison
- Colourful heuristic indicators and chart
- Evidence links and explanation
- Session-only history and CSV reports
- Screenshot OCR using Tesseract
- TF-IDF + Logistic Regression training and evaluation from a labelled CSV

## Important limitation
The live-news indicators are **not truth probabilities**. Google News results may be incomplete, repeated or unrelated. The included sample data only demonstrates the ML workflow; it is not a validated dataset or evidence of real-world accuracy.

## Windows setup
1. Extract this ZIP.
2. Open Command Prompt in the folder containing `app.py`.
3. Run `python -m pip install -r requirements.txt`
4. Run `python -m streamlit run app.py`
5. Open the local URL printed in the terminal (usually `http://localhost:8501`).

For OCR on Windows, install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki, then restart Command Prompt and Streamlit.

## Update your existing GitHub repo
1. Back up your current `truthlens_ai` folder.
2. Copy the new `app.py`, `requirements.txt`, `packages.txt`, `README.md`, `.gitignore`, and `data` folder into your existing project folder. Replace files when prompted.
3. Test locally first.
4. Open your existing `aim509-fake-news-classifier` GitHub repository and select the branch used by your Streamlit deployment.
5. Upload/replace the files. Ensure `data/demo_training_data.csv` is inside the `data` folder in the repository.
6. Commit the changes to that same branch.
7. In Streamlit Community Cloud, confirm repository, branch and main file path. If `app.py` is at repository root, use `app.py`; if inside `truthlens_ai`, use `truthlens_ai/app.py`. Reboot/redeploy if needed.

`packages.txt` asks Streamlit Community Cloud to install the Linux package `tesseract-ocr`.

## Training CSV format
```csv
text,label
"Chandrayaan-3 landed on the Moon in 2023",REAL
"The Moon is made of cheese",FAKE
```
Use a substantial, trusted and documented labelled dataset for meaningful evaluation. The app needs at least 10 usable rows and at least two examples of each class.
