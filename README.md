# Sentiment Analysis with XLNet and BiLSTM

This project performs sentiment analysis by classifying text as positive or negative. It uses advanced natural language processing techniques with the XLNet model and a Bidirectional Long Short-Term Memory (BiLSTM) network. The model is trained on the IMDb movie reviews dataset as well as a custom dataset of Delhi Metro user comments, achieving high accuracy in sentiment classification.

## Requirements
- Python 3.x
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- matplotlib
- seaborn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Directory Structure
```bash
root/
│
├── data/
│   └── delhimetro.csv          # Custom Delhi Metro dataset (contact for access)
│
├── src/
│   ├── model.py                # Model definition (BiLSTM with XLNet)
│   ├── train.py                # Model training script
│   ├── preprocess.py           # Data preprocessing functions
│   └── datasets.py             # Dataset loading and processing
│
├── utils.py                    # Utility functions
├── main.py                     # Entry point for execution
├── requirements.txt            # Required packages
└── .gitignore                  # Git ignore file
```

## Usage
### Data Preparation: 
The Delhi Metro dataset is not publicly available. To access this custom dataset, please contact me directly.

### Run the Model: 
Execute the entire sentiment analysis pipeline by running the following command:
```bash
python main.py
```

### Visualizations: 
The project generates plots for training loss, test accuracy, and confusion matrices, which are saved as PNG files.

## Results
The model provides classification reports and visualizes performance through confusion matrices for both datasets, enabling easy evaluation of model accuracy. The implementation demonstrates high accuracy in sentiment classification, making it effective for real-world applications.

## **Research Paper**

This project is based on our research paper:

**[Sentiment Analysis for Citizen Feedback in Smart Cities with XLNet-BiLSTM: Delhi Metro as a Case Study](https://cse.mait.ac.in/index.php/cse/conference-symposium/international-symposium/105-latest-news/1092-scctt-2024)**  
Published in: *[SCCTT 2024]*  

The paper details the methodology, experiments, and findings, showcasing how the XLNet-BiLSTM model achieves high accuracy for sentiment classification.  
[Read the Full Paper](https://ceur-ws.org/Vol-3868/Paper5.pdf)


## License
This project is licensed under the MIT License.
