import numpy as np
from random import choices, sample, choice
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from generate_data import gen_square,gen_circle, dict_append, sum_params, gen_noise
from models import MLP, CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GENERATING DATA

data = []
img_size = 32
data_size_for_class = 2500
circle_label = torch.tensor([0])
square_label = torch.tensor([1])
# noise_label = torch.tensor([2])
# rectangle_label = torch.tensor([2])
# triangle_label = torch.tensor([3])
num_classes = 2


add_noise = False
random_lables = True

lables = [torch.tensor([0]), torch.tensor([1])]
# for i in range(data_size_for_class):
#     if random_lables:
#         data.append((gen_circle(img_size,choice(range(5,int(img_size*0.5))), add_noise=add_noise),choice(lables)))
#         data.append((gen_square(img_size,choice(range(7,img_size+1)), add_noise=add_noise),choice(lables)))
#     else:
#         data.append((gen_circle(img_size, choice(range(5, int(img_size * 0.5))), add_noise=add_noise), circle_label))
#         data.append((gen_square(img_size, choice(range(7, img_size + 1)), add_noise=add_noise), square_label))
for i in range(data_size_for_class):
    data.append((gen_noise(img_size),choice(lables)))


train_set = data[:int(0.8*data_size_for_class*num_classes)]
test_set = data[int(0.8*data_size_for_class*num_classes):]



# MAIN

mlp_hidden_dim = 400
cnn_hidden_dim = 100

filters = 18
EPOCHS = 30
BATCH_SIZE = 10
res_dict = {}

MLP = MLP(img_size, mlp_hidden_dim).to(device)
CNN = CNN(img_size,cnn_hidden_dim, filters).to(device)

models = {"MLP": MLP, "CNN":CNN}


for model_name, model  in models.items():
    print(model_name)
    sum_params(model)


for model_name, model in models.items():
    print(f"start {model_name}")
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCHS):
        print(f"start epoch {epoch}")
        # TRAIN

        correct = 0
        total = 0
        loss_sum = 0
        shuffeled_train_set = sample(train_set, len(train_set))
        i = 0
        for data_point in shuffeled_train_set:
            i += 1
            label = data_point[1]
            loss, prediction = model(data_point)
            loss_sum += loss
            loss = loss/BATCH_SIZE
            loss.backward()
            if i % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()

            if label == prediction:
                correct += 1
            total += 1
        dict_append(f"{model_name} train_loss_list", float(loss_sum/len(train_set)),res_dict )
        # res_dict["train_loss_list"].append(loss_sum/len(train_set))
        print(f"train accuracy for epoch number {epoch} : {correct / total}")

        # res_dict["train_accuracy_list"].append(correct/total)
        dict_append(f"{model_name} train_accuracy_list", correct/total,res_dict )

        # TEST
        if not random_lables:
            correct = 0
            total = 0
            loss_sum = 0
            confusion_matrix = np.zeros((num_classes,num_classes))

            for data_point in test_set:
                label = int(data_point[1])
                loss, prediction = model(data_point)
                loss_sum += loss
                if label == prediction:
                    correct += 1

                confusion_matrix[label, prediction] += 1
                total += 1

            # res_dict["test_loss_list"].append(loss_sum/len(test_set))
            dict_append(f"{model_name} test_loss_list", float(loss_sum/len(train_set)),res_dict )

            print(f"test accuracy for epoch number {epoch} : {correct / total}")

            # res_dict["test_accuracy_list"].append(correct / total)
            dict_append(f"{model_name} test_accuracy_list", correct/total,res_dict )

    torch.save(model.to('cpu').state_dict(), f"{model_name}.pkl")

    # visualizations

    # plt.plot(res_dict[f"{model_name} train_accuracy_list"], c="blue", label="train Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.plot(res_dict[f"{model_name} test_accuracy_list"], c="red", label="test Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.savefig(f'{model_name} with noise acc-epos.png')
    #
    # plt.clf()

    # plt.plot(res_dict[f"{model_name} train_loss_list"], c="blue", label="train Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Value")
    # plt.legend()
    #
    # plt.plot(res_dict[f"{model_name} test_loss_list"], c="red", label="test Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.savefig(f'{model_name} with noise loss-epochs.png')
    #
    # plt.clf()

    # class_names = ["circle", "square"]
    # fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix,
    #                                 colorbar=True,
    #                                 show_absolute=False,
    #                                 show_normed=True,
    #                                 class_names=class_names)
    #
    # plt.savefig(f'{model_name} with noise conf-mat.png')
    #
    # plt.clf()


print(res_dict)

