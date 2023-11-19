import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

plt.rcParams.update({'font.size': 8})

classes_to_code = {"De Bruços Diagonal": 0,
                   "De Bruços Centralizado": 1,
                   "De Bruços Lateral": 2,
                   "De Bruços Transversal": 3,
                   "De Barriga Para Cima Centralizado": 4,
                   "De Barriga Para Cima Diagonal": 5,
                   "De Barriga Para Cima Lateral": 6,
                   "De Barriga Para Cima Transversal": 7,
                   "De Lado Centralizado": 8,
                   "De Lado Diagonal": 9,
                   "De Lado Lateral": 10,
                   "De Lado Transversal": 11,
                   "Em Pé": 12,
                   "Sentado Centralizado": 13,
                   "Sentado Lateral": 14,
                   "Descendo Da Maca": 15,
                   "Maca Vazia": 16,
                   "Indefinido": 17,
                   "Outros": 18}

code_to_classes = {0: "De Bruços Diagonal",
                   1: "De Bruços Centralizado",
                   2: "De Bruços Lateral",
                   3: "De Bruços Transversal",
                   4: "De Barriga Para Cima Centralizado",
                   5: "De Barriga Para Cima Diagonal",
                   6: "De Barriga Para Cima Lateral",
                   7: "De Barriga Para Cima Transversal",
                   8: "De Lado Centralizado",
                   9: "De Lado Diagonal",
                   10: "De Lado Lateral",
                   11: "De Lado Transversal",
                   12: "Em Pé",
                   13: "Sentado Centralizado",
                   14: "Sentado Lateral",
                   15: "Descendo Da Maca",
                   16: "Maca Vazia",
                   17: "Indefinido",
                   18: "Outros"}


def escrever_lista_csv(nome_arquivo, lista):
    try:
        with open(nome_arquivo, 'w', newline='') as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            for linha in lista:
                escritor_csv.writerow(linha)
        print(f'Lista foi escrita em "{nome_arquivo}" com sucesso.')
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")


def get_model(n_inputs,
              n_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(n_inputs), name='input'))

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer="adam",
                  metrics=['accuracy'])

    model.summary()

    return model


def load_csv(nome_arquivo):
    dados = list()
    with open(nome_arquivo, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dados.append(row)
    return dados


def balance_dataset(classes_to_attributes):
    maxlen = 1000

    # equilibrando o dataset
    for classe in classes_to_attributes:
        class_data = np.array(classes_to_attributes[classe])
        empty_size = maxlen - len(classes_to_attributes[classe])
        empty_slots = empty_size / len(classes_to_attributes[classe])
        if empty_slots >= 1:
            for _ in range(int(empty_slots)):
                classes_to_attributes[classe] += class_data.copy().tolist()
        if len(classes_to_attributes[classe]) < maxlen:
            left_len = maxlen - len(classes_to_attributes[classe])
            classes_to_attributes[classe] += class_data[0:left_len].copy().tolist()
    return classes_to_attributes


def load_cv_dataset(nome_arquivo):
 dados = load_csv(nome_arquivo)
    classes_to_attributes = dict()
    for row in dados:
        classe = row[len(row) - 1]
        if classe not in classes_to_attributes:
            classes_to_attributes[classe] = list()

        classes_to_attributes[classe].append([float(p) for p in row[0:len(row) - 1]])

    classes_to_attributes = balance_dataset(classes_to_attributes)
    data_x = list()
    data_y = list()

    for classe in classes_to_attributes:
        data_x += classes_to_attributes[classe]
        data_y += [encode(classe)] * len(classes_to_attributes[classe])
    return data_x, data_y


def split_attributes_and_classes(nome_arquivo):
    dados = load_csv(nome_arquivo)
    classes_to_attributes = dict()
    for row in dados:
        classe = row[len(row) - 1]
        if classe not in classes_to_attributes:
            classes_to_attributes[classe] = list()

        classes_to_attributes[classe].append([float(p) for p in row[0:len(row) - 1]])

    classes_to_attributes = balance_dataset(classes_to_attributes)
    data_x = list()
    data_y = list()

    # train test split
    for classe in classes_to_attributes:
        data_x += classes_to_attributes[classe]
        data_y += [encode(classe)] * len(classes_to_attributes[classe])

    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=0.5)

    return train_data_x, test_data_x, train_data_y, test_data_y

def encode(classe):
    try:
        return classes_to_code[classe]
    except KeyError as e:
        return -1


def decode(code):
    try:
        return code_to_classes[code]
    except KeyError as e:
        return 'Maca Vazia'


def get_acc(y_true,
            y_pred):
    res = np.zeros(len(y_true))
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            res[i] = 1
    return res


def get_tp(cf):
    tps = list()
    for i in range(len(cf)):
        tps.append(cf[i][i])
    return np.array(tps)


def get_fp(cf):
    fps = list()
    for i in range(len(cf)):
        cfp = 0
        for j in range(len(cf)):
            cfp += cf[j][i] if i != j else 0
        fps.append(cfp)
    return np.array(fps)


def get_fn(cf):
    fps = list()
    for i in range(len(cf)):
        cfp = 0
        for j in range(len(cf)):
            cfp += cf[i][j] if i != j else 0
        fps.append(cfp)
    return np.array(fps)


def get_precision(tp, fp):
    p = tp / (tp + fp)
    return p


def get_recall(tp, fn):
    r = tp / (tp + fn)
    return r


def get_f1_score(p, r):
    f1_score = 2 * (p * r) / (p + r)
    return f1_score


def get_performance_indexes(cf):
    r_classes = [p for p in classes_to_code.keys()]
    ntp = get_tp(cf)
    nfp = get_fp(cf)
    fn = get_fn(cf)
    p = get_precision(ntp, nfp)
    r = get_recall(ntp, fn)
    f1_score = get_f1_score(p, r)
    p *= 100
    r *= 100

    performance = np.stack((ntp,
                            nfp,
                            fn,
                            p,
                            r,
                            f1_score), axis=0).T.copy().tolist()

    f_performance = list()
    for i in range(len(r_classes)):
        performance_row = [r_classes[i]] + ["{:.4f}".format(p) for p in performance[i]]
        f_performance.append(performance_row)

    return f_performance


def get_confusion_matrix(y_true,
                         y_pred,
                         commands,
                         filename):
    plt.close('all')
    test_acc = np.sum(get_acc(y_true, y_pred)) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred).numpy()
    group_counts = ["{0: 0.0f}".format(value) for value in confusion_mtx.flatten()]
    group_percentages = list()
    for i in range(0, len(confusion_mtx)):
        rsum = np.sum(confusion_mtx[i])
        for j in range(0, len(confusion_mtx[i])):
            group_percentages.append("{0:.2f} %".format(100 * confusion_mtx[i][j] / rsum))
    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(len(commands), len(commands))
    plt.figure(figsize=(20, 20))
    sns.heatmap(confusion_mtx,
                xticklabels=commands,
                yticklabels=commands,
                annot=labels,
                fmt='',
                cmap='Blues')
    plt.title(f'Acurácia: {test_acc:.0%}')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.tight_layout()
    plt.savefig(filename)
    return confusion_mtx


def consolidate_cross_validation(cross_validation):
    width, height = cross_validation.shape
    media_colunas = np.zeros(width + 1)
    media_linhas = np.zeros(height)

    # media das linhas
    for i in range(width):
        media_linhas[i] = np.sum(cross_validation[i])/width

    # media das colunas
    for j in range(height):
        media_colunas[j] = np.sum(cross_validation[:,j])/height

    media_colunas[width] = np.sum(media_linhas)/height
    # folds
    folds = np.array([p+1 for p in range(height+1)])

    # empilhando
    cross_validation = np.stack((cross_validation.T, media_linhas), axis=0).T.copy()
    cross_validation = np.stack((cross_validation, media_colunas), axis=0)
    cross_validation = np.stack((folds, cross_validation.T), axis=0).T.copy()

    # não esquecemos do cabeçalho!!!
    header = np.array(['Fold'] + [p for p in classes_to_code.keys()] + ['-'])
    cross_validation = [header] + cross_validation.tolist()
    return cross_validation

#: training procedure

train_data_x, test_data_x, train_data_y, test_data_y = split_attributes_and_classes("pose_classes/poses.csv")

model = get_model(51, 19)
model.fit(train_data_x, train_data_y, batch_size=10, epochs=500, verbose=1)

#: performance

acc = model.evaluate(train_data_x, train_data_y)
print("Loss:", acc[0], " Accuracy:", acc[1])

model.save("models/pose_classifier")

#: tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print('Model size: %dKB' % (len(tflite_model) / 1024))

with open('models/pose_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

#: indexes

pred = model.predict(test_data_x)
pred_y = pred.argmax(axis=-1)

ml_labels = [decode(p) for p in pred_y]
cl_labels = [decode(p) for p in test_data_y]

commands = [p for p in classes_to_code.keys()]

cf = get_confusion_matrix(test_data_y, pred_y, commands, "models/cf_matrix_pose_estimation.png")

performance_indexes = get_performance_indexes(cf)
performance_indexes = [["Classe", "Tp", "Fp", "Fn", "P(%)", "R(%)", "F1 Score"]] + performance_indexes

escrever_lista_csv("pose_classes/performance_indexes_pose_estimation.csv", performance_indexes)

#: cross validation

folds = 10

cross_validation = list()

x, y = load_cv_dataset("pose_classes/poses.csv")

kf = KFold(n_splits=folds)

for train, test in kf.split(x, y):

    train_data_x, test_data_x, train_data_y, test_data_y = x[train], x[test], y[train], y[test]

    model = get_model(51, 19)

    model.fit(train_data_x, train_data_y, batch_size=10, epochs=500, verbose=1)

    #: performance

    acc = model.evaluate(train_data_x, train_data_y)
    print("Loss:", acc[0], " Accuracy:", acc[1])

    #: indexes

    pred = model.predict(test_data_x)
    pred_y = pred.argmax(axis=-1)

    ml_labels = [decode(p) for p in pred_y]
    cl_labels = [decode(p) for p in test_data_y]

    commands = [p for p in classes_to_code.keys()]

    """cf = get_confusion_matrix(test_data_y, pred_y, commands,
                              "models/cf_matrix_pose_estimation_fold_{:s}.png".format(str(fold)))

    performance_indexes = get_performance_indexes(cf)
    performance_indexes = [["Classe", "Tp", "Fp", "Fn", "P(%)", "R(%)", "F1 Score"]] + performance_indexes

    escrever_lista_csv("pose_classes/performance_indexes_pose_estimation_fold_{:s}.csv".format(str(fold)),
                       performance_indexes)"""

    cf = tf.math.confusion_matrix(test_data_y, pred_y).numpy()

    # acumulando os accs
    tp = get_tp(cf)
    fp = get_fp(cf)
    precision = get_precision(tp, fp)
    cross_validation.append(precision.tolist())

# cross_validation = consolidate_cross_validation(cross_validation)

escrever_lista_csv("pose_classes/cross_validacao.csv", cross_validation)
