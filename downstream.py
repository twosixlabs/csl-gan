# https://github.com/reihaneh-torkzadehmahani/DP-CGAN/blob/master/DP_CGAN/dp_conditional_gan_mnist/Base_DP_CGAN.py

from mlxtend.data import loadlocal_mnist
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets

import argparse
from pathlib import Path
import csv

import options
import util

CLASSIFIERS = ["svm", "dt", "lr", "rf", "gnb", "bnb", "ab", "mlp"]

parser = argparse.ArgumentParser()

parser.add_argument("path", type=str, help="Path to the output folder containing the generator save")
parser.add_argument("-e", "--epochs", type=int, default=None, help="Epochs trained for the generator save")
parser.add_argument("-ei", "--epoch_interval", type=int, default=100)
parser.add_argument("-bs", "--batch_size", type=int, default=50)
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-c", "--classifiers", type=str, default=["lr"], nargs="*", choices=CLASSIFIERS)
parser.add_argument("-f", "--folder", default=False, action="store_true")

opt = parser.parse_args()

opt.path = util.add_slash(opt.path)

train_opt = options.load_opt(opt.path + "opt.txt")

torch.set_grad_enabled(False)

if train_opt.dataset != "MNIST":
    raise Exception("Downstream evaluation only implemented for MNIST.")

def compute_fpr_tpr_roc(Y_test, Y_score):
    n_classes = Y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr],
                                                                                       Y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc

@ignore_warnings(category=ConvergenceWarning)
def classify(X_train, Y_train, X_test, classiferName, random_state_value=0):
    if classiferName == "svm":
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state_value))
    elif classiferName == "dt":
        classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state_value))
    elif classiferName == "lr":
        classifier = OneVsRestClassifier(
            LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=random_state_value))
    elif classiferName == "rf":
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state_value))
    elif classiferName == "gnb":
        classifier = OneVsRestClassifier(GaussianNB())
    elif classiferName == "bnb":
        classifier = OneVsRestClassifier(BernoulliNB(alpha=.01))
    elif classiferName == "ab":
        classifier = OneVsRestClassifier(AdaBoostClassifier(random_state=random_state_value))
    elif classiferName == "mlp":
        classifier = OneVsRestClassifier(MLPClassifier(random_state=random_state_value, alpha=1))
    else:
        print("Classifier not in the list!")
        exit()

    Y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    return Y_score


G, _ = util.init_models(train_opt, init_D=False)
G.to(opt.device)
G.eval()

z = torch.empty((10000, train_opt.g_latent_dim), device=opt.device).normal_(0.0, 1.0)
y = torch.empty((10000), device=opt.device).random_(0, 10).long()
zs = torch.split(z, opt.batch_size)
ys = torch.split(F.one_hot(y, num_classes=10).float(), opt.batch_size)
y = y.cpu().numpy()
    
    
# Load MNIST

X_test, Y_test = loadlocal_mnist(
    images_path=train_opt.data_path + 'MNIST/raw/t10k-images-idx3-ubyte',
    labels_path=train_opt.data_path + 'MNIST/raw/t10k-labels-idx1-ubyte')
X_test = X_test.astype(float) / 255.0
Y_test = [int(y) for y in Y_test]

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y_test = label_binarize(Y_test, classes=classes)

log = open(opt.path + "downstream_log.csv", "a")
logger = csv.writer(log)
logger.writerow(["Epoch"] + [c + " AUROC" for c in opt.classifiers])
log.flush()


# Loop through saves with epoch interval

epoch = opt.epoch_interval if opt.epochs is None else opt.epochs
while True:
    path = opt.path + "saves/G-" + str(epoch)
    if not Path(path).is_file():
        break
        
    # Load model
    util.load_model(opt.path + "saves/G-" + str(epoch), G)
        
    # Generate images
    images = None
    for i in range(len(zs)):
        images = G(zs[i], ys[i]) if images is None else torch.cat([images, G(zs[i], ys[i])])
    images = images.reshape(images.size(0), -1).cpu().numpy()
    
    # Run classifiers
    
    aurocs = []
    for c in opt.classifiers:
        Y_score = classify(images, y, X_test, "lr", random_state_value=30)
        _, _, roc_auc = compute_fpr_tpr_roc(Y_test, Y_score)
        print("{} AUROC ({}):  {}".format(c, epoch, roc_auc["micro"]))
        aurocs.append(roc_auc["micro"])
        
    logger.writerow([epoch] + aurocs)
    log.flush()
    
    if opt.epochs is None:
        epoch += opt.epoch_interval
    else:
        break
        
log.close()