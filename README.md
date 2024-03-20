# POS Tagger

### How to run files

#### 1. pos_tagger.py

for Fast Forward Nueral Network
```bash
python pos_tagger.py -f
```


for Recurrent Neural Network
```bash
python pos_tagger.py -r
```
Enter input sentence 

#### FFNN 
For fast forward neural network 
```bash
python ffnn.py 
```

#### RNN 
For fast forward neural network 
```bash
python rnn.py 
```

2. Plotting Graph 
- To plot graph between Accuracy and epochs for RNN Network

```bash
python plotting_graph.py
```

3. Other files

model.pt - A pretrained model made using FFNN Network

model2.pt - A pretrained model made using RNN Network

rnn_acc1.txt - comma seperated values of epochs and Accuracy for configuration 1

rnn_acc2.txt - comma seperated values of epochs and Accuracy for configuration 2

rnn_acc1.txt - comma seperated values of epochs and Accuracy for configuration 3

ffnn_accuracy.txt - dictionary of context_size and accuracy.



