# Consumer Complaint Classification

This project implements a text classification system for consumer complaints using machine learning. The system categorizes complaints into four categories:
- 0: Credit reporting, repair, or other
- 1: Debt collection
- 2: Consumer Loan
- 3: Mortgage

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone or download this repository to your local machine.

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `consumer_complaint_classification.py`: Main script containing the classification implementation
- `requirements.txt`: List of Python dependencies
- `complaints.csv`: Dataset file (not included in repository)

## Dataset

The project expects a CSV file named `complaints.csv` with at least two columns:
- `complaint_text`: The text of the consumer complaint
- `category`: The category label (0, 1, 2, or 3)

## Running the Project

1. Ensure your dataset (`complaints.csv`) is in the same directory as the script.

2. Run the main script:
```bash
python consumer_complaint_classification.py
```

The script will:
- Load and analyze the dataset
- Preprocess the text data
- Train multiple classification models
- Evaluate model performance
- Show the best performing model
- Demonstrate prediction on an example complaint

## Features

- Text preprocessing including:
  - Lowercase conversion
  - Special character removal
  - Stopword removal
  - Lemmatization
- Multiple classification models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
- TF-IDF vectorization
- Model evaluation metrics
- Example prediction functionality

## Output

The script will display:
- Dataset information and statistics
- Model training progress
- Performance metrics for each model
- Best performing model selection
- Example prediction results

## Customization

You can modify the following parameters in the script:
- `max_features` in TfidfVectorizer (currently set to 5000)
- Test size in train_test_split (currently 20%)
- Model parameters in the `train_and_evaluate_models` function

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed:
```bash
pip install -r requirements.txt
```

2. Verify your dataset format matches the expected structure

3. Check that you have sufficient memory for processing large datasets

## License

This project is open source and available under the MIT License. 