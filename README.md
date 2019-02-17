
## Neural Network
### General Idea
As Input, we have the Cosinus/Jaccard Similarity of all possible 13-Gramm Pairs between the Preselection and the text to check.
Our target is to identify which Pair is plagiarism. Therefore we use a Neural Network with two Input neurons (Cosinus/Jaccard Similarity) and two Output neurons (Plagiarism/No plagiarism). To train the Network, we created examples of 13-Gramm Pairs with Cosinus/Jaccard Similarity and labeled which pairs were plagiarism and which were not.

### Data
Data.txt include 120 plagiarism Pairs and 308 Pairs which were not.
Drain: 1/3
DVal: 1/3
DTest: 1/3

### Network architecture
Four fully Connected Layers with two neurons each

activation function hidden: sigmoid
activation function out: softmax
optimizing: backpropagation with Logistic loss function and L-BFGS as optimization routine.

epoch Number = 500

### Evaluation
Accuracy: 0.9097
(tp + tn) / (tp + fp + tn +fn)

Precision: 0.9128
tp / (tp + fp)

Recall: 0.9097
tp / (tp +fn)

tp = True Positive
tn = True Negative
fp = False Positive
fn = False Negative


### Information about Implementation

The Scala Object NN in the file NN.scla loads the Data, create a model, train the model and calculates Accuracy, Precision, Recall Score for DVal or DTest.
