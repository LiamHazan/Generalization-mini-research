import numpy as np
from random import choices, sample, choice
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image


variances = [0.25]

def gen_square(size, edge_len, add_noise = False):
    res = np.zeros((size,size))
    dist_from = choices(range(size-edge_len+1),k=2)
    for i in range(size):
        for j in range(size):
            if i >= dist_from[0] and i < (dist_from[0] + edge_len) and j >= dist_from[1] and j < edge_len + dist_from[1]:
                res[i,j] =1
    # print(dist_from)
    noise = np.random.multivariate_normal(np.full(size,0),np.diag(np.full(size,sample(variances,1))),(size))
    # noise = abs(noise)
    if add_noise:
        return torch.tensor(res + noise, dtype=torch.float)
    return torch.tensor(res, dtype=torch.float)

# a = np.array(gen_square(32,17, add_noise=True))
# print(a)
# im = Image.fromarray(a*255)
# im.show()

def gen_circle(size, radius, add_noise = False):
    res = np.zeros((size,size))
    center = choices(range(round(radius),round(size-radius+1)),k=2)
    for i in range(size):
        for j in range(size):
            if np.sqrt((i-center[0])**2+(j-center[1])**2) < radius:
                res[i,j] = 1
    noise = np.random.multivariate_normal(np.full(size,0),np.diag(np.full(size,sample(variances,1))),(size))
    # noise = abs(noise)
    if add_noise:
        return torch.tensor(res + noise, dtype=torch.float)
    return torch.tensor(res, dtype=torch.float)

# a = np.array(gen_circle(32, 14, add_noise=True))
# print(a)
# im = Image.fromarray(a*255)
# im.show()


def gen_rectangle(size, edge_len1, edge_len2, add_noise = False):
    if edge_len1==edge_len2:
        print("error, this is a square")
    res = np.zeros((size,size))
    max_len = max(edge_len1,edge_len2)
    dist_from1 = sample(range(size-edge_len1+1),1)
    dist_from2 = sample(range(size-edge_len2+1),1)

    for i in range(size):
        for j in range(size):
            if i >= dist_from1[0] and i < (dist_from1[0] + edge_len1) and j >= dist_from2[0] and j < edge_len2 + dist_from2[0]:
                res[i,j] =1
    # print(dist_from)
    noise = np.random.multivariate_normal(np.full(size,0),np.diag(np.full(size,sample(variances,1))),(size))
    # noise = abs(noise)
    if add_noise:
        return torch.tensor(res + noise, dtype=torch.float)
    return torch.tensor(res, dtype=torch.float)



def gen_triangle(size, base_len, add_noise = False):
    res = np.zeros((size,size))
    height_len = round(base_len/2)
    height = choice(range(height_len-1,size+1))
    dist_from_left = choice(range(size-base_len+1))

    for i in range(size):
        for j in range(size):
            dist_from_base = height - i
            if j >= dist_from_left+dist_from_base and j < dist_from_left+base_len - dist_from_base and i <= height and i > height-height_len:
                res[i,j] =1
    # print(dist_from)
    noise = np.random.multivariate_normal(np.full(size,0),np.diag(np.full(size,sample(variances,1))),(size))
    # noise = abs(noise)
    if add_noise:
        return torch.tensor(res + noise, dtype=torch.float)
    return torch.tensor(res, dtype=torch.float)
#
# a = np.array(gen_triangle(10, 7, add_noise=True))
# print(a)
# im = Image.fromarray(a*255)
# im.show()


def gen_noise(size):
    noise = np.random.multivariate_normal(np.full(size,0),np.diag(np.full(size,sample(variances,1))),(size))
    return torch.tensor(noise, dtype=torch.float)



def dict_append(key, val, dictionary):
    """
        Implements the dictionary insertion for the 'find_absolute_tags' function
        :param key: dictionary key
        :param dictionary: the dictionary created in the external function

    """
    if key not in dictionary:
        dictionary[key] = [val]
    else:
        dictionary[key].append(val)


def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def sum_params(model):
    sum = 0
    for name, params in model.state_dict().items():
        if multiplyList(params.shape) != 1:
            print(name, multiplyList(params.shape))
            sum += multiplyList(params.shape)
    print(f"sum :{sum}")


#
# data = []
# data_size = 10000
# stders = [1,2,3]
# for i in range(data_size):
#     stder = sample(stders,1)
#     x = np.random.normal(0,stder,144)
#     stder = stder[0]
#     data.append((x.reshape((12,12)),stder))
#
# correct = 0
# total = 0
#
# for data_point in data:
#     x, label = data_point[0], data_point[1]
#     # if (label -2.5) <= x.std() <= (label+2.5):
#     if round(x.std()) == label:
#         correct += 1
#     total += 1
#
# print(f"the accuracy is: {correct/total}")
#
