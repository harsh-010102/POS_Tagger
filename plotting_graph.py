import matplotlib.pyplot as plt

x = []
y = []
# c1
# embedding_dim = 200
# num_layers = 2
# bidirectional = True
# hidden_dim = 10
# epoch = 10
# c2
# embedding_dim = 400
# num_layers = 3
# bidirectional = True
# hidden_dim = 20
# epoch = 10

with open('rnn_acc_c3.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        x.append(int(parts[0])) 
        y.append(float(parts[1]))

plt.plot(x, y, marker='o')  # Plot with data points marked as 'o'
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('For configuration : Embedding Dim = 200, Num of Layer = 4,\n Hidden Dim = 10, Biderectional = True')
plt.grid(True)
plt.show()
