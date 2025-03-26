## Document Classification for QTLdb Curation

This project aims to automate the identification of research papers relevant to the Animal QTLdb, a database that curates genotype-to-phenotype relationships in livestock. Using natural language processing and a fine-tuned BERT model, the goal is to classify scientific papers based on their relevance to QTLdb curation.

By accurately filtering relevant documents, the model supports human curators in accelerating the review process, ensuring that important genotype-trait findings are not overlooked. The project emphasizes high recall to minimize the risk of missing valuable papers, even at the cost of including some irrelevant ones.

> **Note:** For privacy and size considerations, the training data and model checkpoints have been excluded from this repository.


## Training and Inferencing Notebooks
Training and inferencing notebooks used for this project are in the `bert_method` folder.

## Report & Analysis
For a detailed report on findings and term analysis, refer to:
[Phrase Mining Report (PDF)](report/NLP_AnimalQTL_Document_Classification.pdf). 
