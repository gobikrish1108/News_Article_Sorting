
# News Article Sorting Project using LSTM

## Project Overview:
This project aims to classify news articles into one of five categories: **Sport**, **Business**, **Politics**, **Entertainment**, and **Tech** using a **Long Short-Term Memory (LSTM)** neural network. We used a dataset containing 1490 news articles to train the model and achieved an impressive **93.96% test accuracy**.

## Dataset Details:
- **Total Records**: 1490 articles
- **Categories**: Sport (346), Business (336), Politics (274), Entertainment (273), Tech (261)
- **Features**: `ArticleId`, `Text`, `Category`, `News_length`, `Text_parsed`, `Category_target`

## Model Details:
We built an LSTM model using the following layers:
1. **Embedding Layer**: Converts word indices into dense vectors of fixed size.
2. **LSTM Layers**: Two LSTM layers with 64 units each, followed by dropout layers to prevent overfitting.
3. **Dense Layers**: Three fully connected layers with ReLU activation, followed by dropout.
4. **Output Layer**: A softmax layer for multi-class classification with 5 categories.

### Model Training:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Validation Split**: 20% of the training data
- **Early Stopping**: Used to stop training when the validation loss does not improve for 7 epochs (with a minimum delta of 0.01).

### Model Evaluation:
- **Test Accuracy**: **93.96%**
