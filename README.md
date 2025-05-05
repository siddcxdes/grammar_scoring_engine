# Grammar Scoring Engine Project

This project is a **Grammar Scoring Engine** that processes audio files, transcribes them to text, and evaluates their grammar. It uses both machine learning and deep learning methods to score the quality of grammar in audio transcriptions. Below is a summary of the steps and methodology:

---

## Steps in the Pipeline

1. **Audio Pre-Processing**:
   - Audio files are preprocessed using the `librosa` library by resampling, trimming noise, and normalizing the volume.

2. **Transcription**:
   - Preprocessed audio files are transcribed into text using OpenAI's `Whisper` model.

3. **Combining with Scores**:
   - Transcriptions are combined with `train.csv` to create a dataset for further use.

4. **Cleaning the Transcriptions**:
   - The transcription text is cleaned using regular expressions to remove filler words, non-ASCII characters, and unwanted spaces.

5. **Grammar Checking**:
   - Grammar mistakes in the cleaned transcriptions are counted using the `language_tool_python` library.

6. **Feature Engineering**:
   - Additional features like **parts of speech diversity**, **sentence length**, and **stopword ratio** are generated using `spaCy`.

7. **Model Training**:
   - Two models are trained:
     - **Random Forest Regressor**: Trained on engineered features.
     - **DistilBERT**: Extracts sentence embeddings and trains a Ridge regression model.

8. **Hybrid Ensembling**:
   - A **meta-model** is created using the predictions from both Random Forest and DistilBERT. This final model uses K-fold cross-validation and Ridge regression for better accuracy.

---

## Evaluation Metrics (Meta-Model with K-Fold on RF + BERT)

The meta-model was evaluated on test data using the following metrics:
- **Mean Absolute Error (MAE)**: `0.7096`
- **Mean Squared Error (MSE)**: `0.7845`
- **Root Mean Squared Error (RMSE)**: `0.8857`
- **Pearson Correlation**: `0.6699`

---

## Conclusion

This project successfully combines audio preprocessing, transcription, grammar checking, and machine learning to create a scoring engine. The hybrid model, which uses both Random Forest and DistilBERT, achieves a strong correlation (`0.6699`) between predicted and true scores, making it effective for grammar evaluation tasks.
