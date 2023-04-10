import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from multiprocessing import Process

def load_pickle(pickle_path: str):

    with open(pickle_path, "rb") as fd:
        return pickle.load(fd)


def save_pickle(obj: object, pickle_path: str):

    with open(pickle_path, "wb") as fd:
        pickle.dump(obj, fd)


def load_train_test_idxs(idxs_dir: str) -> tuple:

    train_idxs = load_pickle(f"{idxs_dir}/train.pkl")
    test_idxs = load_pickle(f"{idxs_dir}/test.pkl")
    return train_idxs, test_idxs


# Make dict of words to document indexes.


def get_word_to_index(df_train: pd.DataFrame, top_words_idf: dict) -> dict:

    word_to_idxs = {}
    for sample in df_train.itertuples():
        for word in set(sample.text.split(' ')):
            if word in top_words_idf:
                if word not in word_to_idxs:
                    word_to_idxs[word] = []
                word_to_idxs[word].append(sample.idx)

    word_to_idxs = {word: set(word_to_idxs[word]) for word in word_to_idxs}
    return word_to_idxs

# Make dict of labels to document indexes.


def get_label_to_index(df_train: pd.DataFrame) -> dict:

    label_to_idxs = {}
    for sample in df_train.itertuples():
        for label in sample.labels:
            if label not in label_to_idxs:
                label_to_idxs[label] = []
            label_to_idxs[label].append(sample.idx)

    label_to_idxs = {label: set(label_to_idxs[label])
                     for label in label_to_idxs}
    return label_to_idxs

# Computing words probabilities - P(w).


def get_p_w(df_train: pd.DataFrame,
            top_words_idf: dict,
            n_docs: int) -> dict:

    p_w = {}
    for sample in df_train.itertuples():
        for word in set(sample.text.split(' ')):
            if word in top_words_idf:
                if word not in p_w:
                    p_w[word] = 0
                p_w[word] += 1

    p_w = {word: p_w[word] / n_docs for word in p_w}
    return p_w

# Computing labels probabilities - P(l).


def get_p_l(df_train: pd.DataFrame, n_docs: int) -> dict:

    p_l = {}
    for sample in df_train.itertuples():
        for label in sample.labels:
            if label not in p_l:
                p_l[label] = 0
            p_l[label] += 1

    p_l = {label: p_l[label] / n_docs for label in p_l}
    return p_l


def get_pmi(
        word: str,
        label: str,
        p_w: dict,
        p_l: dict,
        word_to_idxs: dict,
        label_to_idxs: dict,
        n_docs: int
) -> tuple:

    pwl = len(word_to_idxs[word].intersection(label_to_idxs[label])) / n_docs
    pw = p_w[word]
    pl = p_l[label]

    if pw == 0 or pl == 0:
        return 0

    return np.log10(pwl / (pw * pl)), pwl


def get_p_wl_pmi(word_to_idxs: dict,
                 label_to_idxs: dict,
                 p_w: dict,
                 p_l: dict,
                 word_max_pmi_path: str,
                 p_wl_path,
                 n_docs,
                 load_precomputed: bool) -> tuple:

    if load_precomputed \
        and os.path.exists(word_max_pmi_path) \
            and os.path.exists(p_wl_path):

        word_max_pmi = load_pickle(word_max_pmi_path)
        p_wl = load_pickle(p_wl_path)

        return word_max_pmi, p_wl

    p_wl = {}
    word_max_pmi = {word: 0 for word in word_to_idxs }

    for word in tqdm(p_w.keys(), mininterval=10):
        for label in p_l.keys():
            pmi_value, pwl = get_pmi(
                word, label, p_w, p_l, word_to_idxs, label_to_idxs, n_docs)
            if pmi_value > 0:
                p_wl[f"{word}_{label}"] = pwl
            if pmi_value > word_max_pmi[word]:
                word_max_pmi[word] = pmi_value

    # Saving precomputed pickles.
    save_pickle(word_max_pmi, word_max_pmi_path)
    save_pickle(p_wl, p_wl_path)

    return word_max_pmi, p_wl


def get_apmi(
        word: str,
        label: str,
        p_w: dict,
        p_l: dict,
        p_wl: dict
) -> tuple:

    key = f"{word}_{label}"
    pwl = p_wl[key] if key in p_wl else 0
    
    if word not in p_w:
        return 0
    
    pw = p_w[word]
    pl = p_l[label]

    if pl == 0 or pw == 0:
        return 0

    return np.log10((pwl * pwl) / (pw * pl))


# Compute P(w,l) and also the max pmi for each word.
# You can use load_precomputed to load the precomputed dicts
# instead of create it from scratch.


def generate_test(df_test: pd.DataFrame,
                  p_w: dict,
                  p_l: dict,
                  p_wl: dict,
                  word_max_pmi: dict,
                  labels_ids: dict,
                  label_cls: dict,
                  ) -> dict:

    test_data = {
        "all": {},
        "tail": {},
        "head": {}
    }

    for sample in tqdm(df_test.itertuples(), total=df_test.shape[0], mininterval=10, desc="Computing aPMI"):

        text_id = f"text_{sample.text_idx}"
        max_pmi = 0
        top_word = ''
        
        # Getting the word with the biggest PMI in the document.
        for word in set(sample.text.split(' ')):
            if word in word_max_pmi:
                if word_max_pmi[word] > max_pmi:
                    top_word = word
                    max_pmi = word_max_pmi[word]

        # Computing the aPMI for word & labels in the document..
        labels_apmi = [
            [label, get_apmi(top_word, label, p_w, p_l, p_wl)]
            for label in p_l
        ]

        # Distributing the documents in classes by label.
        for label_tup in labels_apmi:
            lidx = labels_ids[label_tup[0]]
            for cls in label_cls[lidx]:
                if text_id not in test_data[cls]:
                    test_data[cls][text_id] = {}
                label_key = f"label_{lidx}"
                test_data[cls][text_id][label_key] = label_tup[1]

        # Getting the 128 labels with bigger apmi.
        # For each label type.
        for cls in test_data:
            # For each document in each label type.
            for text_id in test_data[cls]:
                # Sortind labels by apmi.
                ranking = [[label, test_data[cls][text_id][label]]
                           for label in test_data[cls][text_id]
                           ]
                
                ranking.sort(key=lambda x: x[1], reverse=True)

                test_data[cls][text_id] = {l[0]: l[1] for l in ranking[:128]}

    return test_data



def build(
    dataset: str,
    fold: str,
    df: pd.DataFrame,
    topn: int,
    labels_ids: dict,
    label_cls: dict
):
    
    test_data = {}
    
    print(f"DATASET: {dataset} / FOLD: {fold}")
    train_idxs, test_idxs = load_train_test_idxs(
        f"data/{dataset}/fold_{fold}/")
    df_train = df[df.idx.isin(train_idxs)]#.sample(n=1000, random_state=42)
    df_test = df[df.idx.isin(test_idxs)]#.sample(n=1000, random_state=42)

    n_docs = df_train.shape[0]

    # ## Applying TF-IDF and selecting the TOP50K highest IDF words.
    tf = TfidfVectorizer(max_features=topn)
    tf.fit(df_train.text.tolist())
    top_words_idf = {word: tf.idf_[tf.vocabulary_[word]]
                        for word in tf.get_feature_names_out()}

    print("\n\tComputing p_w and p_l...")
    p_w = get_p_w(df_train, top_words_idf, n_docs)
    p_l = get_p_l(df_train, n_docs)

    print("\n\tComputing word's max pmi and p_wl...")

    word_to_idxs = get_word_to_index(df_train, top_words_idf)
    label_to_idxs = get_label_to_index(df_train)

    word_max_pmi_path = f"data/{dataset}/fold_{fold}/word_max_pmi.pkl"
    p_wl_path = f"data/{dataset}/fold_{fold}/p_wl.pkl"

    word_max_pmi, p_wl = get_p_wl_pmi(word_to_idxs,
                                        label_to_idxs,
                                        p_w,
                                        p_l,
                                        word_max_pmi_path,
                                        p_wl_path,
                                        n_docs,
                                        True)

    print("\n\tGenarating test estimations...")
    test_data[fold] = generate_test(df_test,
                                    p_w,
                                    p_l,
                                    p_wl,
                                    word_max_pmi,
                                    labels_ids,
                                    label_cls)
    save_pickle(p_wl, f"data/{dataset}/fold_{fold}/p_wl.pkl")
    save_pickle(word_max_pmi, f"data/{dataset}/fold_{fold}/word_max_pmi.pkl")

    save_pickle(test_data, f"data/{dataset}/fold_{fold}/test_data.pkl")

if __name__=='__main__':

    with open("data/params_execution.json", 'r') as fd:
        params_execution = json.load(fd)

    dataset = params_execution["DATASET"]
    topn = params_execution["TOPN"]
    FOLDS = params_execution["FOLDS"]


    # Loading document samples.
    samples = load_pickle(f"data/{dataset}/samples.pkl")
    df = pd.DataFrame(samples)
    label_cls = load_pickle(f"data/{dataset}/label_cls.pkl")
    labels_ids = {}
    for row in df.itertuples():
        for idx in range(len(row.labels)):
            label = row.labels[idx]
            lid = row.labels_ids[idx]
            if label not in labels_ids:
                labels_ids[label] = lid

    process = []
    for fold in FOLDS:

        p = Process(
            target = build,
            args = (
                dataset,
                fold,
                df,
                topn,
                labels_ids,
                label_cls,))

        p.start()
        process.append(p)

    for p in process:
        p.join()
    
    test_data = {}
    for fold in np.arange(0, len(FOLDS)):
        test_file = f"data/{dataset}/fold_{fold}/test_data.pkl"
        if os.path.exists(test_file):
            test_data = load_pickle(test_file)
            test_data.update(load_pickle(test_file))
    
    save_pickle(test_data, f"data/{dataset}/APMI_{dataset}.rnk")