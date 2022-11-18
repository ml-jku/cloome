import sys
import os
import json
import logging
from pathlib import Path
from typing import Tuple, Union, List
from collections import OrderedDict
from statistics import mode


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import nn

import clip.clip as clip
from training.datasets import CellPainting
from clip.clip import _transform
from clip.model import convert_weights, CLIPGeneral

import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-path",
        type=str,
        default=None,
        help="Path to csv index file",
    )
    parser.add_argument(
        "--train-imgs",
        type=str,
        default=None,
        help="Path to csv filewith training",
    )
    parser.add_argument(
        "--val-imgs",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--test-imgs",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--row-index-file",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--col-index-file",
        type=str,
        default=None,
        help="Path to csv file with validation",
    )
    parser.add_argument(
        "--image-resolution",
        nargs='+',
        type=int,
        help="In DP, which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--model",
        choices=["RN50", "RN101", "RN50x4", "ViT-B/32"],
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument("--seed", default=1234, type=int, help="Seed for reproducibility")
    args = parser.parse_args()
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_features(dataset, model, device, seed):
    g = torch.Generator()
    g.manual_seed(seed)

    all_features = []
    all_labels = []
    print(f"get_features {device}")
    print(len(dataset))
    with torch.no_grad():
        for input_dict in tqdm(DataLoader(dataset, num_workers=15, batch_size=64, worker_init_fn=seed_worker, generator=g)):
            images = input_dict["input"]
            labels = input_dict["target"]
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)


    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def load(model_path, device, args):
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict["state_dict"]

    model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIPGeneral(**model_info)
    convert_weights(model)

    if str(device) == "cpu":
        model.float()
    print(device)

    new_state_dict = {k[7:]: v for k,v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, _transform(args.image_resolution, args.image_resolution, is_train=False)


def initialize(device, args):

    model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)

    model = CLIPGeneral(**model_info)
    model.to(device)
    model.eval()

    return model, _transform(args.image_resolution, args.image_resolution, is_train=False)


def get_metrics(classifier, test_features, test_labels, where_test_labels, i):
    predictions = classifier.predict(test_features[where_test_labels])
    probs = classifier.predict_proba(test_features[where_test_labels])

    target = test_labels[where_test_labels, i]

    accuracy = np.mean((target == predictions).astype(np.float))
    auc = roc_auc_score(y_true=target, y_score=probs[:, 1])
    mcc = matthews_corrcoef(y_true=target, y_pred=predictions)
    f1 = f1_score(y_true=target, y_pred=predictions)

    results = {"AUC" : auc,
              "Accuracy": accuracy,
              "MCC": mcc,
              "F1": f1
    }

    return results


def main(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = args.model_path
    log_path = args.log_path
    img_path = args.img_path
    train_imgs = args.train_imgs
    val_imgs = args.val_imgs
    test_imgs = args.test_imgs
    device = args.device
    matrix_file = args.matrix_file
    row_index_file = args.row_index_file
    col_index_file = args.col_index_file
    image_resolution = args.image_resolution


    p = Path(model_path)
    exp_name = p.parts[-3]
    metrics_path = os.path.join(log_path, exp_name)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if not os.path.exists(metrics_path):
        os.mkdir(metrics_path)

    tb_path = os.path.join(metrics_path, "tensorboard")

    writer = SummaryWriter(tb_path)


    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load(model_path, device, args)

    preprocess_train = _transform(image_resolution, image_resolution, is_train=True)
    preprocess_val = _transform(image_resolution, image_resolution, is_train=False)


    # Load the dataset
    train = CellPainting(train_imgs,
                        img_path,
                        label_matrix_file=matrix_file,
                        label_row_index_file=row_index_file,
                        label_col_index_file=col_index_file,
                        transforms = preprocess_train)


    val = CellPainting(val_imgs,
                       img_path,
                       label_matrix_file=matrix_file,
                       label_row_index_file=row_index_file,
                       label_col_index_file=col_index_file,
                       transforms = preprocess_val)


    test = CellPainting(test_imgs,
                        img_path,
                        label_matrix_file=matrix_file,
                        label_row_index_file=row_index_file,
                        label_col_index_file=col_index_file,
                        transforms = preprocess_val)

    # Calculate the image features
    print("train")
    train_features, train_labels = get_features(train, model, device, args.seed)
    print(f"train_labels: {train_labels.shape}")
    print("val")
    val_features, val_labels = get_features(val, model, device, args.seed)
    print(f"val_labels: {val_labels.shape}")
    print("test")
    test_features, test_labels = get_features(test, model, device, args.seed)
    print(f"test_labels: {test_labels.shape}")

    n_tasks = train.num_classes
    print(n_tasks)


    class_aucs, class_accs, class_mccs, class_f1s = [], [], [], []
    test_aucs, test_accs, test_mccs, test_f1s = [], [], [], []
    best_Cs = []


    columns = ["N_actives_train", "N_inactives_train", "N_labels_train", \
               "N_actives_val", "N_inactives_val", "N_labels_val", \
               "N_actives_test", "N_inactives_test", "N_labels_test", \
               "Val_AUC", "Val_Accuracy", "Val_MCC", "Val_F1", \
               "Test_AUC", "Test_Accuracy", "Test_MCC", "Test_F1", "Best_C",]

    df = pd.DataFrame(columns=columns)

    C_param_range = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10**2, 10**3, 10**4, 10**5, 10**6]
    best_results = {}

    for i in range(n_tasks):
        train_actives = list(np.where(train_labels[:, i] == 1.)[0])
        train_inactives = list(np.where(train_labels[:, i] == -1.)[0])
        where_train_labels = train_actives + train_inactives

        n_train_actives = len(train_actives)
        n_train_inactives = len(train_inactives)
        n_train_labels = n_train_actives + n_train_inactives

        test_actives = list(np.where(test_labels[:, i] == 1.)[0])
        test_inactives = list(np.where(test_labels[:, i] == -1.)[0])
        where_test_labels = test_actives + test_inactives

        n_test_actives = len(test_actives)
        n_test_inactives = len(test_inactives)
        n_test_labels = n_test_actives + n_test_inactives

        val_actives = list(np.where(val_labels[:, i] == 1.)[0])
        val_inactives = list(np.where(val_labels[:, i] == -1.)[0])
        where_val_labels = val_actives + val_inactives

        n_val_actives = len(val_actives)
        n_val_inactives = len(val_inactives)
        n_val_labels = n_val_actives + n_val_inactives


        if train_actives and train_inactives \
        and val_actives and val_inactives:

            best_auc = 0

            for C in C_param_range:
                # Perform logistic regression
                classifier = LogisticRegression(random_state=0, C=C, max_iter=3000)
                classifier.fit(train_features[where_train_labels], train_labels[where_train_labels, i])

                val_metrics = get_metrics(classifier, val_features, val_labels, where_val_labels, i)

                train_metrics = get_metrics(classifier, train_features, train_labels, where_train_labels, i)

                writer.add_scalar(f"Tasks/Task_{i}/Val", val_metrics["AUC"], C)

                if val_metrics["AUC"] > best_auc:
                    best_auc = val_metrics["AUC"]
                    best_results = val_metrics
                    best_classifier = classifier
                    best_C = C

            val_auc = best_results["AUC"]
            val_accuracy = best_results["Accuracy"]
            val_mcc = best_results["MCC"]
            val_f1 = best_results["F1"]

            train_val_features = np.concatenate([train_features, val_features])
            train_val_labels = np.concatenate([train_labels, val_labels])


            #where_val_labels_cat = where_val_labels + len(where_train_labels)
            where_val_labels_cat = [where_lab + train_labels.shape[0] for where_lab in where_val_labels]


            where_train_val_labels = np.concatenate([where_train_labels, where_val_labels_cat])

            best_classifier.fit(train_val_features[where_train_val_labels], train_val_labels[where_train_val_labels, i])


            if test_actives and test_inactives:
                test_metrics = get_metrics(best_classifier, test_features, test_labels, where_test_labels, i)

                test_auc = test_metrics["AUC"]
                test_accuracy = test_metrics["Accuracy"]
                test_mcc = test_metrics["MCC"]
                test_f1 = test_metrics["F1"]
            else:
                print(f"Missing test in {i}")
                test_auc = 0.5
                test_accuracy = 0
                test_mcc = -1
                test_f1 = 0

        else:
            print(f"Missing in {i}")
            val_auc = 0.5
            val_accuracy = 0
            val_mcc = -1
            val_f1 = 0

            if test_actives and test_inactives:
                if best_Cs:
                    C = mode(best_Cs)
                else:
                    C = 1
                    best_C = 99

                classifier = LogisticRegression(random_state=0, C=C, max_iter=3000)
                classifier.fit(train_features[where_train_labels], train_labels[where_train_labels, i])

                train_metrics = get_metrics(classifier, train_features, train_labels, where_train_labels, i)
                test_metrics = get_metrics(classifier, test_features, test_labels, where_test_labels, i)

                test_auc = test_metrics["AUC"]
                test_accuracy = test_metrics["Accuracy"]
                test_mcc = test_metrics["MCC"]
                test_f1 = test_metrics["F1"]


        df.loc[i] = [n_train_actives, n_train_inactives, n_train_labels, \
        n_val_actives, n_val_inactives, n_val_labels, \
        n_test_actives, n_test_inactives, n_test_labels, \
        val_auc, val_accuracy, val_mcc, val_f1, \
        test_auc, test_accuracy, test_mcc, test_f1, best_C \
        ]

        class_aucs.append(val_auc)
        class_accs.append(val_accuracy)
        class_mccs.append(val_mcc)
        class_f1s.append(val_f1)

        test_aucs.append(test_auc)
        test_accs.append(test_accuracy)
        test_mccs.append(test_mcc)
        test_f1s.append(test_f1)

        best_Cs.append(best_C)

    mean_auc = float(np.mean(class_aucs))
    mean_acc = float(np.mean(class_accs))
    mean_mcc = float(np.mean(class_mccs))
    mean_f1 = float(np.mean(class_f1s))

    test_mean_auc = float(np.mean(test_aucs))
    test_mean_acc = float(np.mean(test_accs))
    test_mean_mcc = float(np.mean(test_mccs))
    test_mean_f1 = float(np.mean(test_f1s))

    df.loc[i+1, "Val_AUC"] = mean_auc
    df.loc[i+1, "Val_Accuracy"] = mean_acc
    df.loc[i+1, "Val_MCC"] = mean_mcc
    df.loc[i+1, "Val_F1"] = mean_f1

    df.loc[i+1, "Test_AUC"] = test_mean_auc
    df.loc[i+1, "Test_Accuracy"] = test_mean_acc
    df.loc[i+1, "Test_MCC"] = test_mean_mcc
    df.loc[i+1, "Test_F1"] = test_mean_f1


    f = Path(p.name)
    f = f.with_suffix('.csv')

    path = os.path.join(metrics_path, f)

    if os.path.exists(path):
        basename, ext = os.path.splitext(path)
        new_basename = f"{basename}_1"
        path = new_basename + ext

    df.to_csv(path)

    print(' *AUC {auc:.3f}'.format(auc=mean_auc))
    return mean_auc


if __name__ == '__main__':
    args = parse_args()
    main(args)
