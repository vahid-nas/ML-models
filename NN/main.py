from No_Hidden_NN import NN_LinearRegression
from Data_Preprocessing import Data_Prep
import matplotlib.pyplot as plt
from One_Hidden_NN import ANN

#Data preprocessing part
data = Data_Prep(Normalize=True)
X_train1 , y_train1, X_test1, y_test1, X_train2, y_train2,X_test2, y_test2 = data.spdata()


# model = NN_LinearRegression()
# model.train(X_train1,y_train1)
# loss = model.loss(X_test1,y_test1)
# print("Test Loss:  "+str(loss))
# model.plot_loss()
# text = "No hidden layer NN - 1st dataset - results on Train set"
# model.plot_scatter(X_train1,y_train1,text)
# text = "No hidden layer NN - 1st dataset - results on Test set"
# model.plot_scatter(X_test1,y_test1,text)
# loss = model.loss(X_test1,y_test1)
# print("Test Loss:  "+str(loss))
#
#
# model = NN_LinearRegression()
# model.train(X_train2,y_train2)
# loss = model.loss(X_test1,y_test1)
# print("Test Loss:  "+str(loss))
# model.plot_loss()
# text = "No hidden layer NN - 2st dataset - results on Train set"
# model.plot_scatter(X_train2,y_train2,text)
# text = "No hidden layer NN - 2st dataset - results on Test set"
# model.plot_scatter(X_test2,y_test2,text)


# model2 = ANN(Loss_threshold=0,number_of_nodes_in_hidden_layer=4)
# model2.train(X_train1,y_train1)
# # model2.train_stochastic(X_train1,y_train1)
# model2.plot_loss()
# text = "One hidden layer NN - 1st dataset - results on Train set"
# model2.plot_scatter(X_train1,y_train1,text)
# text = "One hidden layer NN - 1st dataset - results on Test set"
# model2.plot_scatter(X_test1,y_test1,text)
#
#
model2 = ANN(Loss_threshold=0,momentum=True,number_of_nodes_in_hidden_layer=8,learning_rate=0.025,n_epochs = 1000000)
# model2.train(X_train2,y_train2)
model2.train_stochastic(X_train1,y_train1)
model2.plot_loss()
text = "One hidden layer NN - 2st dataset - results on Train set"
model2.plot_scatter(X_train2,y_train2,text)
text = "One hidden layer NN - 2st dataset - results on Test set"
model2.plot_scatter(X_test2,y_test2,text)







