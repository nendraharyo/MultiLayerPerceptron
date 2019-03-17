# import dataframe and plot library
import pandas as pd
import matplotlib.pyplot as plt
import math

# define table formatting, classify dataset class
pd.options.display.max_columns = None
cols = ['x1', 'x2', 'x3', 'x4', 'class', 'target1', 'target2', 'prediction1', 'prediction2', 'wi1', 'wi2', 'wi3', 'wi4',
        'wi5', 'wi6', 'wi7', 'wi8', 'biashid1', 'biashid2', 'si1', 'si2', 'wh1', 'wh2', 'wh3', 'wh4', 'biasout1',
        'biasout2', 'soutput1', 'soutput2', 'error1', 'error2', 'totalerror', 'dwo1', 'dwo2', 'dwo3', 'dwo4', 'dwi1',
        'dwi2', 'dwi3', 'dwi4', 'dwi5', 'dwi6', 'dwi7', 'dwi8']
df = pd.read_csv("iris.data", header=None, names=cols)
df.head(150)

for i in range(150):
    if df.at[i, 'class'] == 'Iris-setosa':
        df.at[i, 'target1'] = 0
        df.at[i, 'target2'] = 0
    elif df.at[i, 'class'] == 'Iris-versicolor':
        df.at[i, 'target1'] = 0
        df.at[i, 'target2'] = 1
    elif df.at[i, 'class'] == 'Iris-virginica':
        df.at[i, 'target1'] = 1
        df.at[i, 'target2'] = 1

# split the data to 30:120
df_validation = df.iloc[0:10]
df_validation = df_validation.append(df.iloc[50:60], ignore_index=True, sort=False)
df_validation = df_validation.append(df.iloc[100:110], ignore_index=True, sort=False)

df_training = df.iloc[10:50]
df_training = df_training.append(df.iloc[60:100], ignore_index=True, sort=False)
df_training = df_training.append(df.iloc[110:150], ignore_index=True, sort=False)

# input learning rate, epoch, and prepare epoch data
lrate = float(input("learning rate: "))
epoch = int(input("epoch: "))

# random init and first value for first training data
df_training.at[0, 'wi1'] = df_training.at[0, 'wi2'] = df_training.at[0, 'wi3'] = df_training.at[0, 'wi4'] = \
df_training.at[0, 'wi5'] = df_training.at[0, 'wi6'] = df_training.at[0, 'wi7'] = df_training.at[0, 'wi8'] = \
df_training.at[0, 'biashid1'] = df_training.at[0, 'biashid2'] = df_training.at[0, 'wh1'] = df_training.at[0, 'wh2'] = \
df_training.at[0, 'wh3'] = df_training.at[0, 'wh4'] = df_training.at[0, 'biasout1'] = df_training.at[
    0, 'biasout2'] = 0.5

# activate hidden layer
df_training.at[0, 'si1'] = 1 / (1 + math.exp(-(
            df_training.at[0, 'x1'] * df_training.at[0, 'wi1'] + df_training.at[0, 'x2'] * df_training.at[0, 'wi2'] +
            df_training.at[0, 'x3'] * df_training.at[0, 'wi3'] + df_training.at[0, 'x4'] * df_training.at[0, 'wi4'] +
            df_training.at[0, 'biashid1'])))
df_training.at[0, 'si2'] = 1 / (1 + math.exp(-(
            df_training.at[0, 'x1'] * df_training.at[0, 'wi5'] + df_training.at[0, 'x2'] * df_training.at[0, 'wi6'] +
            df_training.at[0, 'x3'] * df_training.at[0, 'wi7'] + df_training.at[0, 'x4'] * df_training.at[0, 'wi8'] +
            df_training.at[0, 'biashid2'])))

# activate output
df_training.at[0, 'soutput1'] = 1 / (1 + math.exp(-(
            df_training.at[0, 'si1'] * df_training.at[0, 'wh1'] + df_training.at[0, 'wh2'] * df_training.at[0, 'si2'] +
            df_training.at[0, 'biasout1'])))
df_training.at[0, 'soutput2'] = 1 / (1 + math.exp(-(
            df_training.at[0, 'si1'] * df_training.at[0, 'wh3'] + df_training.at[0, 'wh4'] * df_training.at[0, 'si2'] +
            df_training.at[0, 'biasout2'])))

# get error
df_training.at[0, 'error1'] = math.pow(df_training.at[0, 'target1'] - df_training.at[0, 'soutput1'], 2) / 2
df_training.at[0, 'error2'] = math.pow(df_training.at[0, 'target2'] - df_training.at[0, 'soutput2'], 2) / 2
df_training.at[0, 'totalerror'] = df_training.at[0, 'error1'] + df_training.at[0, 'error2']

# prediction
df_training.at[0, 'prediction1'] = round(df_training.at[0, 'soutput1'])
df_training.at[0, 'prediction2'] = round(df_training.at[0, 'soutput2'])

# backprop output-hiddenlayer
df_training.at[0, 'dwo1'] = (df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'soutput1'] * (1 - df_training.at[0, 'soutput1']) * df_training.at[0, 'si1']
df_training.at[0, 'dwo2'] = (df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'soutput1'] * (1 - df_training.at[0, 'soutput1']) * df_training.at[0, 'si2']
df_training.at[0, 'dwo3'] = (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[
    0, 'soutput2'] * (1 - df_training.at[0, 'soutput2']) * df_training.at[0, 'si1']
df_training.at[0, 'dwo4'] = (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[
    0, 'soutput2'] * (1 - df_training.at[0, 'soutput2']) * df_training.at[0, 'si2']

# backprop hiddenlayer-input
df_training.at[0, 'dwi1'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh1'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh3']) * \
                            df_training.at[0, 'si1'] * (1 - df_training.at[0, 'si1']) * df_training.at[0, 'x1']
df_training.at[0, 'dwi2'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh1'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh3']) * \
                            df_training.at[0, 'si1'] * (1 - df_training.at[0, 'si1']) ** df_training.at[0, 'x2']
df_training.at[0, 'dwi3'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh1'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh3']) * \
                            df_training.at[0, 'si1'] * (1 - df_training.at[0, 'si1']) ** df_training.at[0, 'x3']
df_training.at[0, 'dwi4'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh1'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh3']) * \
                            df_training.at[0, 'si1'] * (1 - df_training.at[0, 'si1']) ** df_training.at[0, 'x4']

df_training.at[0, 'dwi5'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh2'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh4']) * \
                            df_training.at[0, 'si2'] * (1 - df_training.at[0, 'si2']) * df_training.at[0, 'x1']
df_training.at[0, 'dwi6'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh2'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh4']) * \
                            df_training.at[0, 'si2'] * (1 - df_training.at[0, 'si2']) ** df_training.at[0, 'x2']
df_training.at[0, 'dwi7'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh2'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh4']) * \
                            df_training.at[0, 'si2'] * (1 - df_training.at[0, 'si2']) ** df_training.at[0, 'x3']
df_training.at[0, 'dwi8'] = ((df_training.at[0, 'soutput1'] - df_training.at[0, 'target1']) * df_training.at[
    0, 'wh2'] + (df_training.at[0, 'soutput2'] - df_training.at[0, 'target2']) * df_training.at[0, 'wh4']) * \
                            df_training.at[0, 'si2'] * (1 - df_training.at[0, 'si2']) ** df_training.at[0, 'x4']

df_training.head()

# start the loop!
ploterrortrain = []
ploterrorvalidate = []
plotcorrecttrain = []
plotcorrectvalidate = []
plotcounter = []
countercorrect = 0
countercorrectval = 0

for j in range(1, epoch):
    plotcounter.append(j)

    for i in range(1, df_training.shape[0]):
        # update wh1 - wh4
        df_training.at[i, 'wh1'] = df_training.at[i - 1, 'wh1'] - lrate * df_training.at[i - 1, 'dwo1']
        df_training.at[i, 'wh2'] = df_training.at[i - 1, 'wh1'] - lrate * df_training.at[i - 1, 'dwo2']
        df_training.at[i, 'wh3'] = df_training.at[i - 1, 'wh1'] - lrate * df_training.at[i - 1, 'dwo3']
        df_training.at[i, 'wh4'] = df_training.at[i - 1, 'wh1'] - lrate * df_training.at[i - 1, 'dwo4']

        # update wi1 - wi8
        df_training.at[i, 'wi1'] = df_training.at[i - 1, 'wi1'] - lrate * df_training.at[i - 1, 'dwi1']
        df_training.at[i, 'wi2'] = df_training.at[i - 1, 'wi2'] - lrate * df_training.at[i - 1, 'dwi2']
        df_training.at[i, 'wi3'] = df_training.at[i - 1, 'wi3'] - lrate * df_training.at[i - 1, 'dwi3']
        df_training.at[i, 'wi4'] = df_training.at[i - 1, 'wi4'] - lrate * df_training.at[i - 1, 'dwi4']
        df_training.at[i, 'wi5'] = df_training.at[i - 1, 'wi5'] - lrate * df_training.at[i - 1, 'dwi5']
        df_training.at[i, 'wi6'] = df_training.at[i - 1, 'wi6'] - lrate * df_training.at[i - 1, 'dwi6']
        df_training.at[i, 'wi7'] = df_training.at[i - 1, 'wi7'] - lrate * df_training.at[i - 1, 'dwi7']
        df_training.at[i, 'wi8'] = df_training.at[i - 1, 'wi8'] - lrate * df_training.at[i - 1, 'dwi8']

        # update biases
        df_training.at[i, 'biashid1'] = df_training.at[i - 1, 'biashid1'] - lrate * (((df_training.at[
                                                                                           i - 1, 'soutput1'] -
                                                                                       df_training.at[
                                                                                           i - 1, 'target1']) *
                                                                                      df_training.at[i - 1, 'wh1'] + (
                                                                                                  df_training.at[
                                                                                                      i - 1, 'soutput2'] -
                                                                                                  df_training.at[
                                                                                                      i - 1, 'target2']) *
                                                                                      df_training.at[i - 1, 'wh3']) *
                                                                                     df_training.at[i - 1, 'si1'] * (
                                                                                                 1 - df_training.at[
                                                                                             i - 1, 'si1']))
        df_training.at[i, 'biashid2'] = df_training.at[i - 1, 'biashid2'] - lrate * (((df_training.at[
                                                                                           i - 1, 'soutput1'] -
                                                                                       df_training.at[
                                                                                           i - 1, 'target1']) *
                                                                                      df_training.at[i - 1, 'wh2'] + (
                                                                                                  df_training.at[
                                                                                                      i - 1, 'soutput2'] -
                                                                                                  df_training.at[
                                                                                                      i - 1, 'target2']) *
                                                                                      df_training.at[i - 1, 'wh4']) *
                                                                                     df_training.at[i - 1, 'si2'] * (
                                                                                                 1 - df_training.at[
                                                                                             i - 1, 'si2']))

        df_training.at[i, 'biasout1'] = df_training.at[i - 1, 'biasout1'] - lrate * (((df_training.at[
                                                                                           i - 1, 'soutput1'] -
                                                                                       df_training.at[
                                                                                           i - 1, 'target1']) *
                                                                                      df_training.at[i - 1, 'wh1'] + (
                                                                                                  df_training.at[
                                                                                                      i - 1, 'soutput2'] -
                                                                                                  df_training.at[
                                                                                                      i - 1, 'target2']) *
                                                                                      df_training.at[i - 1, 'wh3']) *
                                                                                     df_training.at[i - 1, 'si1'] * (
                                                                                                 1 - df_training.at[
                                                                                             i - 1, 'si1']))
        df_training.at[i, 'biasout2'] = df_training.at[i - 1, 'biasout2'] - lrate * (((df_training.at[
                                                                                           i - 1, 'soutput1'] -
                                                                                       df_training.at[
                                                                                           i - 1, 'target1']) *
                                                                                      df_training.at[i - 1, 'wh2'] + (
                                                                                                  df_training.at[
                                                                                                      i - 1, 'soutput2'] -
                                                                                                  df_training.at[
                                                                                                      i - 1, 'target2']) *
                                                                                      df_training.at[i - 1, 'wh4']) *
                                                                                     df_training.at[i - 1, 'si2'] * (
                                                                                                 1 - df_training.at[
                                                                                             i - 1, 'si2']))

        # activate hidden layer
        df_training.at[i, 'si1'] = 1 / (1 + math.exp(-(
                    df_training.at[i, 'x1'] * df_training.at[i, 'wi1'] + df_training.at[i, 'x2'] * df_training.at[
                i, 'wi2'] + df_training.at[i, 'x3'] * df_training.at[i, 'wi3'] + df_training.at[i, 'x4'] *
                    df_training.at[i, 'wi4'] + df_training.at[i, 'biashid1'])))
        df_training.at[i, 'si2'] = 1 / (1 + math.exp(-(
                    df_training.at[i, 'x1'] * df_training.at[i, 'wi5'] + df_training.at[i, 'x2'] * df_training.at[
                i, 'wi6'] + df_training.at[i, 'x3'] * df_training.at[i, 'wi7'] + df_training.at[i, 'x4'] *
                    df_training.at[i, 'wi8'] + df_training.at[i, 'biashid2'])))

        # activate output
        df_training.at[i, 'soutput1'] = 1 / (1 + math.exp(-(
                    df_training.at[i, 'si1'] * df_training.at[i, 'wh1'] + df_training.at[i, 'wh2'] * df_training.at[
                i, 'si2'] + df_training.at[i, 'biasout1'])))
        df_training.at[i, 'soutput2'] = 1 / (1 + math.exp(-(
                    df_training.at[i, 'si1'] * df_training.at[i, 'wh3'] + df_training.at[i, 'wh4'] * df_training.at[
                i, 'si2'] + df_training.at[i, 'biasout2'])))

        # get error
        df_training.at[i, 'error1'] = math.pow(df_training.at[i, 'target1'] - df_training.at[i, 'soutput1'], 2) / 2
        df_training.at[i, 'error2'] = math.pow(df_training.at[i, 'target2'] - df_training.at[i, 'soutput2'], 2) / 2
        df_training.at[i, 'totalerror'] = df_training.at[i, 'error1'] + df_training.at[i, 'error2']

        # prediction
        df_training.at[i, 'prediction1'] = round(df_training.at[i, 'soutput1'])
        df_training.at[i, 'prediction2'] = round(df_training.at[i, 'soutput2'])

        if df_training.at[i, 'target1'] == df_training.at[i, 'prediction1'] and df_training.at[i, 'target2'] == \
                df_training.at[i, 'prediction2']:
            countercorrect = countercorrect + 1

        # backprop output-hiddenlayer
        df_training.at[i, 'dwo1'] = (df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'soutput1'] * (1 - df_training.at[i, 'soutput1']) * df_training.at[i, 'si1']
        df_training.at[i, 'dwo2'] = (df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'soutput1'] * (1 - df_training.at[i, 'soutput1']) * df_training.at[i, 'si2']
        df_training.at[i, 'dwo3'] = (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[
            i, 'soutput2'] * (1 - df_training.at[i, 'soutput2']) * df_training.at[i, 'si1']
        df_training.at[i, 'dwo4'] = (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[
            i, 'soutput2'] * (1 - df_training.at[i, 'soutput2']) * df_training.at[i, 'si2']

        # backprop hiddenlayer-input
        df_training.at[i, 'dwi1'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh1'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh3']) * \
                                    df_training.at[i, 'si1'] * (1 - df_training.at[i, 'si1']) * df_training.at[i, 'x1']
        df_training.at[i, 'dwi2'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh1'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh3']) * \
                                    df_training.at[i, 'si1'] * (1 - df_training.at[i, 'si1']) * df_training.at[i, 'x2']
        df_training.at[i, 'dwi3'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh1'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh3']) * \
                                    df_training.at[i, 'si1'] * (1 - df_training.at[i, 'si1']) * df_training.at[i, 'x3']
        df_training.at[i, 'dwi4'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh1'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh3']) * \
                                    df_training.at[i, 'si1'] * (1 - df_training.at[i, 'si1']) * df_training.at[i, 'x4']

        df_training.at[i, 'dwi5'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh2'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh4']) * \
                                    df_training.at[i, 'si2'] * (1 - df_training.at[i, 'si2']) * df_training.at[i, 'x1']
        df_training.at[i, 'dwi6'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh2'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh4']) * \
                                    df_training.at[i, 'si2'] * (1 - df_training.at[i, 'si2']) * df_training.at[i, 'x2']
        df_training.at[i, 'dwi7'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh2'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh4']) * \
                                    df_training.at[i, 'si2'] * (1 - df_training.at[i, 'si2']) * df_training.at[i, 'x3']
        df_training.at[i, 'dwi8'] = ((df_training.at[i, 'soutput1'] - df_training.at[i, 'target1']) * df_training.at[
            i, 'wh2'] + (df_training.at[i, 'soutput2'] - df_training.at[i, 'target2']) * df_training.at[i, 'wh4']) * \
                                    df_training.at[i, 'si2'] * (1 - df_training.at[i, 'si2']) * df_training.at[i, 'x4']

        if i == 119:
            df_training.at[0, 'prediction1'] = df_training.at[119, 'prediction1']
            df_training.at[0, 'prediction2'] = df_training.at[119, 'prediction2']
            df_training.at[0, 'wi1'] = df_training.at[119, 'wi1']
            df_training.at[0, 'wi2'] = df_training.at[119, 'wi2']
            df_training.at[0, 'wi3'] = df_training.at[119, 'wi3']
            df_training.at[0, 'wi4'] = df_training.at[119, 'wi4']
            df_training.at[0, 'wi5'] = df_training.at[119, 'wi5']
            df_training.at[0, 'wi6'] = df_training.at[119, 'wi6']
            df_training.at[0, 'wi7'] = df_training.at[119, 'wi7']
            df_training.at[0, 'wi8'] = df_training.at[119, 'wi8']
            df_training.at[0, 'biashid1'] = df_training.at[119, 'biashid1']
            df_training.at[0, 'biashid2'] = df_training.at[119, 'biashid2']
            df_training.at[0, 'si1'] = df_training.at[119, 'si1']
            df_training.at[0, 'si2'] = df_training.at[119, 'si2']
            df_training.at[0, 'wh1'] = df_training.at[119, 'wh1']
            df_training.at[0, 'wh2'] = df_training.at[119, 'wh2']
            df_training.at[0, 'wh3'] = df_training.at[119, 'wh3']
            df_training.at[0, 'wh4'] = df_training.at[119, 'wh4']
            df_training.at[0, 'biasout1'] = df_training.at[119, 'biasout1']
            df_training.at[0, 'biasout2'] = df_training.at[119, 'biasout2']
            df_training.at[0, 'soutput1'] = df_training.at[119, 'soutput1']
            df_training.at[0, 'soutput2'] = df_training.at[119, 'soutput2']
            df_training.at[0, 'error1'] = df_training.at[119, 'error1']
            df_training.at[0, 'error2'] = df_training.at[119, 'error2']
            df_training.at[0, 'totalerror'] = df_training.at[119, 'totalerror']

            ploterrortrain.append(math.log(df_training.at[i, 'totalerror']))
            plotcorrecttrain.append(countercorrect / 119)
            countercorrect = 0

            # validating
            df_validation.at[0, 'prediction1'] = df_training.at[119, 'prediction1']
            df_validation.at[0, 'prediction2'] = df_training.at[119, 'prediction2']
            df_validation.at[0, 'wi1'] = df_training.at[119, 'wi1']
            df_validation.at[0, 'wi2'] = df_training.at[119, 'wi2']
            df_validation.at[0, 'wi3'] = df_training.at[119, 'wi3']
            df_validation.at[0, 'wi4'] = df_training.at[119, 'wi4']
            df_validation.at[0, 'wi5'] = df_training.at[119, 'wi5']
            df_validation.at[0, 'wi6'] = df_training.at[119, 'wi6']
            df_validation.at[0, 'wi7'] = df_training.at[119, 'wi7']
            df_validation.at[0, 'wi8'] = df_training.at[119, 'wi8']
            df_validation.at[0, 'biashid1'] = df_training.at[119, 'biashid1']
            df_validation.at[0, 'biashid2'] = df_training.at[119, 'biashid2']
            df_validation.at[0, 'si1'] = df_training.at[119, 'si1']
            df_validation.at[0, 'si2'] = df_training.at[119, 'si2']
            df_validation.at[0, 'wh1'] = df_training.at[119, 'wh1']
            df_validation.at[0, 'wh2'] = df_training.at[119, 'wh2']
            df_validation.at[0, 'wh3'] = df_training.at[119, 'wh3']
            df_validation.at[0, 'wh4'] = df_training.at[119, 'wh4']
            df_validation.at[0, 'biasout1'] = df_training.at[119, 'biasout1']
            df_validation.at[0, 'biasout2'] = df_training.at[119, 'biasout2']
            df_validation.at[0, 'soutput1'] = df_training.at[119, 'soutput1']
            df_validation.at[0, 'soutput2'] = df_training.at[119, 'soutput2']
            df_validation.at[0, 'error1'] = df_training.at[119, 'error1']
            df_validation.at[0, 'error2'] = df_training.at[119, 'error2']
            df_validation.at[0, 'totalerror'] = df_training.at[119, 'totalerror']

            df_validation.at[0, 'dwo1'] = df_training.at[119, 'dwo1']
            df_validation.at[0, 'dwo2'] = df_training.at[119, 'dwo2']
            df_validation.at[0, 'dwo3'] = df_training.at[119, 'dwo3']
            df_validation.at[0, 'dwo4'] = df_training.at[119, 'dwo4']

            df_validation.at[0, 'dwi1'] = df_training.at[119, 'dwi1']
            df_validation.at[0, 'dwi2'] = df_training.at[119, 'dwi2']
            df_validation.at[0, 'dwi3'] = df_training.at[119, 'dwi3']
            df_validation.at[0, 'dwi4'] = df_training.at[119, 'dwi4']
            df_validation.at[0, 'dwi5'] = df_training.at[119, 'dwi5']
            df_validation.at[0, 'dwi6'] = df_training.at[119, 'dwi6']
            df_validation.at[0, 'dwi7'] = df_training.at[119, 'dwi7']
            df_validation.at[0, 'dwi8'] = df_training.at[119, 'dwi8']

            for k in range(1, df_validation.shape[0]):
                # update wh1 - wh4
                df_validation.at[k, 'wh1'] = df_validation.at[k - 1, 'wh1'] - lrate * df_validation.at[k - 1, 'dwo1']
                df_validation.at[k, 'wh2'] = df_validation.at[k - 1, 'wh1'] - lrate * df_validation.at[k - 1, 'dwo2']
                df_validation.at[k, 'wh3'] = df_validation.at[k - 1, 'wh1'] - lrate * df_validation.at[k - 1, 'dwo3']
                df_validation.at[k, 'wh4'] = df_validation.at[k - 1, 'wh1'] - lrate * df_validation.at[k - 1, 'dwo4']

                # update wi1 - wi8
                df_validation.at[k, 'wi1'] = df_validation.at[k - 1, 'wi1'] - lrate * df_validation.at[k - 1, 'dwi1']
                df_validation.at[k, 'wi2'] = df_validation.at[k - 1, 'wi2'] - lrate * df_validation.at[k - 1, 'dwi2']
                df_validation.at[k, 'wi3'] = df_validation.at[k - 1, 'wi3'] - lrate * df_validation.at[k - 1, 'dwi3']
                df_validation.at[k, 'wi4'] = df_validation.at[k - 1, 'wi4'] - lrate * df_validation.at[k - 1, 'dwi4']
                df_validation.at[k, 'wi5'] = df_validation.at[k - 1, 'wi5'] - lrate * df_validation.at[k - 1, 'dwi5']
                df_validation.at[k, 'wi6'] = df_validation.at[k - 1, 'wi6'] - lrate * df_validation.at[k - 1, 'dwi6']
                df_validation.at[k, 'wi7'] = df_validation.at[k - 1, 'wi7'] - lrate * df_validation.at[k - 1, 'dwi7']
                df_validation.at[k, 'wi8'] = df_validation.at[k - 1, 'wi8'] - lrate * df_validation.at[k - 1, 'dwi8']

                # update biases
                df_validation.at[k, 'biashid1'] = df_validation.at[k - 1, 'biashid1'] - lrate * (((df_validation.at[
                                                                                                       k - 1, 'soutput1'] -
                                                                                                   df_validation.at[
                                                                                                       k - 1, 'target1']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh1'] + (
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'soutput2'] -
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'target2']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh3']) *
                                                                                                 df_validation.at[
                                                                                                     k - 1, 'si1'] * (
                                                                                                             1 -
                                                                                                             df_validation.at[
                                                                                                                 k - 1, 'si1']))
                df_validation.at[k, 'biashid2'] = df_validation.at[k - 1, 'biashid2'] - lrate * (((df_validation.at[
                                                                                                       k - 1, 'soutput1'] -
                                                                                                   df_validation.at[
                                                                                                       k - 1, 'target1']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh2'] + (
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'soutput2'] -
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'target2']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh4']) *
                                                                                                 df_validation.at[
                                                                                                     k - 1, 'si2'] * (
                                                                                                             1 -
                                                                                                             df_validation.at[
                                                                                                                 k - 1, 'si2']))

                df_validation.at[k, 'biasout1'] = df_validation.at[k - 1, 'biasout1'] - lrate * (((df_validation.at[
                                                                                                       k - 1, 'soutput1'] -
                                                                                                   df_validation.at[
                                                                                                       k - 1, 'target1']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh1'] + (
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'soutput2'] -
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'target2']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh3']) *
                                                                                                 df_validation.at[
                                                                                                     k - 1, 'si1'] * (
                                                                                                             1 -
                                                                                                             df_validation.at[
                                                                                                                 k - 1, 'si1']))
                df_validation.at[k, 'biasout2'] = df_validation.at[k - 1, 'biasout2'] - lrate * (((df_validation.at[
                                                                                                       k - 1, 'soutput1'] -
                                                                                                   df_validation.at[
                                                                                                       k - 1, 'target1']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh2'] + (
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'soutput2'] -
                                                                                                              df_validation.at[
                                                                                                                  k - 1, 'target2']) *
                                                                                                  df_validation.at[
                                                                                                      k - 1, 'wh4']) *
                                                                                                 df_validation.at[
                                                                                                     k - 1, 'si2'] * (
                                                                                                             1 -
                                                                                                             df_validation.at[
                                                                                                                 k - 1, 'si2']))

                # activate hidden layer
                df_validation.at[k, 'si1'] = 1 / (1 + math.exp(-(
                            df_validation.at[k, 'x1'] * df_validation.at[k, 'wi1'] + df_validation.at[k, 'x2'] *
                            df_validation.at[k, 'wi2'] + df_validation.at[k, 'x3'] * df_validation.at[k, 'wi3'] +
                            df_validation.at[k, 'x4'] * df_validation.at[k, 'wi4'] + df_validation.at[k, 'biashid1'])))
                df_validation.at[k, 'si2'] = 1 / (1 + math.exp(-(
                            df_validation.at[k, 'x1'] * df_validation.at[k, 'wi5'] + df_validation.at[k, 'x2'] *
                            df_validation.at[k, 'wi6'] + df_validation.at[k, 'x3'] * df_validation.at[k, 'wi7'] +
                            df_validation.at[k, 'x4'] * df_validation.at[k, 'wi8'] + df_validation.at[k, 'biashid2'])))

                # activate output
                df_validation.at[k, 'soutput1'] = 1 / (1 + math.exp(-(
                            df_validation.at[k, 'si1'] * df_validation.at[k, 'wh1'] + df_validation.at[k, 'wh2'] *
                            df_validation.at[k, 'si2'] + df_validation.at[k, 'biasout1'])))
                df_validation.at[k, 'soutput2'] = 1 / (1 + math.exp(-(
                            df_validation.at[k, 'si1'] * df_validation.at[k, 'wh3'] + df_validation.at[k, 'wh4'] *
                            df_validation.at[k, 'si2'] + df_validation.at[k, 'biasout2'])))

                # get error
                df_validation.at[k, 'error1'] = math.pow(
                    df_validation.at[k, 'target1'] - df_validation.at[k, 'soutput1'], 2) / 2
                df_validation.at[k, 'error2'] = math.pow(
                    df_validation.at[k, 'target2'] - df_validation.at[k, 'soutput2'], 2) / 2
                df_validation.at[k, 'totalerror'] = df_validation.at[k, 'error1'] + df_validation.at[k, 'error2']

                # prediction
                df_validation.at[k, 'prediction1'] = round(df_validation.at[k, 'soutput1'])
                df_validation.at[k, 'prediction2'] = round(df_validation.at[k, 'soutput2'])

                if df_validation.at[k, 'target1'] == df_validation.at[k, 'prediction1'] and df_validation.at[
                    k, 'target2'] == df_validation.at[k, 'prediction2']:
                    countercorrectval = countercorrectval + 1

                # backprop output-hiddenlayer
                df_validation.at[k, 'dwo1'] = (df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) * \
                                              df_validation.at[k, 'soutput1'] * (1 - df_validation.at[k, 'soutput1']) * \
                                              df_validation.at[k, 'si1']
                df_validation.at[k, 'dwo2'] = (df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) * \
                                              df_validation.at[k, 'soutput1'] * (1 - df_validation.at[k, 'soutput1']) * \
                                              df_validation.at[k, 'si2']
                df_validation.at[k, 'dwo3'] = (df_validation.at[k, 'soutput2'] - df_validation.at[k, 'target2']) * \
                                              df_validation.at[k, 'soutput2'] * (1 - df_validation.at[k, 'soutput2']) * \
                                              df_validation.at[k, 'si1']
                df_validation.at[k, 'dwo4'] = (df_validation.at[k, 'soutput2'] - df_validation.at[k, 'target2']) * \
                                              df_validation.at[k, 'soutput2'] * (1 - df_validation.at[k, 'soutput2']) * \
                                              df_validation.at[k, 'si2']

                # backprop hiddenlayer-input
                df_validation.at[k, 'dwi1'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh1'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh3']) * df_validation.at[
                                                  k, 'si1'] * (1 - df_validation.at[k, 'si1']) * df_validation.at[
                                                  k, 'x1']
                df_validation.at[k, 'dwi2'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh1'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh3']) * df_validation.at[
                                                  k, 'si1'] * (1 - df_validation.at[k, 'si1']) * df_validation.at[
                                                  k, 'x2']
                df_validation.at[k, 'dwi3'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh1'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh3']) * df_validation.at[
                                                  k, 'si1'] * (1 - df_validation.at[k, 'si1']) * df_validation.at[
                                                  k, 'x3']
                df_validation.at[k, 'dwi4'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh1'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh3']) * df_validation.at[
                                                  k, 'si1'] * (1 - df_validation.at[k, 'si1']) * df_validation.at[
                                                  k, 'x4']

                df_validation.at[k, 'dwi5'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh2'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh4']) * df_validation.at[
                                                  k, 'si2'] * (1 - df_validation.at[k, 'si2']) * df_validation.at[
                                                  k, 'x1']
                df_validation.at[k, 'dwi6'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh2'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh4']) * df_validation.at[
                                                  k, 'si2'] * (1 - df_validation.at[k, 'si2']) * df_validation.at[
                                                  k, 'x2']
                df_validation.at[k, 'dwi7'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh2'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh4']) * df_validation.at[
                                                  k, 'si2'] * (1 - df_validation.at[k, 'si2']) * df_validation.at[
                                                  k, 'x3']
                df_validation.at[k, 'dwi8'] = ((df_validation.at[k, 'soutput1'] - df_validation.at[k, 'target1']) *
                                               df_validation.at[k, 'wh2'] + (
                                                           df_validation.at[k, 'soutput2'] - df_validation.at[
                                                       k, 'target2']) * df_validation.at[k, 'wh4']) * df_validation.at[
                                                  k, 'si2'] * (1 - df_validation.at[k, 'si2']) * df_validation.at[
                                                  k, 'x4']

            ploterrorvalidate.append(math.log(df_validation.at[29, 'totalerror']))
            plotcorrectvalidate.append(countercorrectval / 30)
            countercorrectval = 0
df_validation.head(3)

# draw the graph
fig, plotaccuracy = plt.subplots()
fig, ploterror = plt.subplots()

len(plotcorrecttrain)
plotaccuracy.set_title("Accuracy Diagram")
plotaccuracy.plot(plotcounter, plotcorrecttrain, color="red", label="accuracy training")
plotaccuracy.plot(plotcounter, plotcorrectvalidate, color="green", label="accuracy training")
plotaccuracy.set_xlabel("number of epochs")
plotaccuracy.set_ylabel("accuracy percentage")
plotaccuracy.legend(loc="upper right")
plt.show()

ploterror.set_title("Error Diagram")
ploterror.plot(plotcounter, ploterrortrain, color="red", label="error training")
ploterror.plot(plotcounter, ploterrorvalidate, color="green", label="error validate")
ploterror.set_xlabel("number of epochs")
ploterror.set_ylabel("log error")
ploterror.legend(loc="upper right")
plt.show()