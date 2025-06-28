# Steam Game Reviews Sentiment Classification

This repository contains a Python project developed for a Natural Language Processing (NLP) mid-semester assessment. It implements and compares three distinct NLP approaches – Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word2Vec – to classify the sentiment of user reviews from the Steam video game platform.

The project is implemented in a Jupyter Notebook and processes a subset of the Steam Review Dataset (2017) to evaluate the models' capability in understanding informal, sarcastic, and domain-specific language common to game reviews.

    Project Tasks & Features:

        - Dataset Preparation and Preprocessing:

          * Uses the Steam Review Dataset (2017) containing over 6.4 million reviews.

        - Focuses on review text and binary sentiment labels (Recommended/Not Recommended).

        - Preprocessing pipeline includes:

            * Removal of URLs, HTML tags, numbers, punctuation, and special characters.

            * Lowercasing and tokenisation.

            * Stop-word removal and lemmatisation for semantic clarity.

        - Post-cleaning, empty reviews are filtered out to maintain dataset integrity.

        - Handling Class Imbalance:

          * Applies Random Oversampling to balance positive and negative classes without discarding valuable data.

        - Model Implementations:

            * Bag-of-Words (BoW): Uses CountVectorizer with Multinomial Naive Bayes classifier.

            * TF-IDF: Uses TfidfVectorizer with Multinomial Naive Bayes classifier.

            * Word2Vec: Uses Gensim's Word2Vec embedding with Logistic Regression classifier. Embeddings trained on tokenised reviews with dimension size 300 and window size 5.

        - Evaluation and Analysis:

          * 80/20 train-test split due to computational constraints.

        - Performance evaluated via:

            * Accuracy

            * Precision, Recall, F1-Score (especially for the minority Negative class)

            * Confusion Matrices

        - Qualitative Error Analysis conducted on 22 challenging reviews featuring sarcasm, slang, and complex negation to compare model robustness.

    Key Results:

    BoW achieved highest accuracy (~82%) but underperformed in identifying negative reviews.

    TF-IDF and Word2Vec models achieved ~72% accuracy but showed significantly better recall for the Negative class (~86%).

    Word2Vec demonstrated strengths in semantic understanding, particularly sarcasm detection, albeit with computational cost.

    Technologies Used:

    Python 3

    Pandas, NumPy, Scikit-learn

    NLTK (for tokenisation and lemmatisation)

    Gensim (Word2Vec embeddings)

    Imbalanced-learn (RandomOverSampler)

    Matplotlib, Seaborn (visualisation)

    Tabulate (for summarised table outputs)

    File Structure:

        actual_midsem_ipynb_report_22062025.ipynb: The main Jupyter Notebook implementing the project.

        archive/dataset_edited.csv: The cleaned dataset input.

        archive/steam_reviews_cleaned.csv: Saved cleaned dataset after preprocessing.

        archive/selected_reviews.csv: The curated set of reviews used for qualitative error analysis.

        requirements.txt: Python dependencies.

    Running the Project:

        Install dependencies:

pip install -r requirements.txt

Ensure dataset files are in the archive folder as per notebook path usage.
