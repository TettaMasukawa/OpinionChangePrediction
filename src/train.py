import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, accuracy_score
from transformers import AutoTokenizer
import itertools
# import wandb
import numpy as np
import pickle

from dataset import Dataset


feature_file = "random-BERT"
# feature_file = "random-JTweetRoBERTa"
# feature_file = "random-SimCSE"
# feature_file = "topk-StyleSimCSE"

option = ""
# option = "_reposts_embedding"
# option = "_two_embedding"
# option = "_two_2"
# option = "_three"
model_name = feature_file.split("_")[-1] + option
print(model_name)

dataset = Dataset(
    f"data/features/oc_features_tweets_{feature_file}.json",
    f"data/features/onc_features_tweets_{feature_file}.json",
    f"data/features/random_features_tweets_{feature_file}.json"
)

dataset_cv_train, tokenized_cv_train, dataset_train, tokenized_train, dataset_valid, tokenized_valid, test_features, test_labels, tokenized_test = dataset.create()

tokenizer = AutoTokenizer.from_pretrained(
    "tohoku-nlp/bert-base-japanese-v3",
    mecab_kwargs={"mecab_dic": None, "mecab_option": "-d {0}".format("/usr/local/lib/mecab/dic/mecab-ipadic-neologd")}
)

# tokenizer = AutoTokenizer.from_pretrained(
#     "MU-Kindai/Japanese-SimCSE-BERT-base-sup",
#     mecab_kwargs={"mecab_dic": None, "mecab_option": "-d {0}".format("/usr/local/lib/mecab/dic/mecab-ipadic-neologd")}
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     "/home/anakada/style_sentence_vector/saved_model/twitter_sup_simcse_base_hard_negative_100per/",
#     mecab_kwargs={"mecab_dic": None, "mecab_option": "-d {0}".format("/usr/local/lib/mecab/dic/mecab-ipadic-neologd")}
# )

# tokenizer = AutoTokenizer.from_pretrained("/home/rtakasu/sotuken/pre_trained_/until_250_len/checkpoint-550000/")

# posts_decoded_train = []
# posts_decoded_valid = []
posts_decoded_test = []
reposts_decoded_test = []
flag_posts_decoded_test = []
posts_decoded_test_split = []
reposts_decoded_test_split = []
flag_posts_decoded_test_split = []

# for i in range(len(posts_tokenized_train)):
#     token_ids = [id for id in posts_tokenized_train["posts_tokenized"][i]]
#     posts_decoded_train.append(tokenizer.decode(token_ids))

# posts_tokenized_train["posts_decoded"] = posts_decoded_train
# posts_tokenized_train.to_csv("results/train_sentences.csv", index=False, encoding="utf-8", columns=["label", "posts_decoded"])

# for i in range(len(posts_tokenized_valid)):
#     token_ids = [id for id in posts_tokenized_valid["posts_tokenized"][i]]
#     posts_decoded_valid.append(tokenizer.decode(token_ids))

# posts_tokenized_valid["posts_decoded"] = posts_decoded_valid
# posts_tokenized_valid.to_csv("results/valid_sentences.csv", index=False, encoding="utf-8", columns=["label", "posts_decoded"])

config = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "max_depth": -1,
    "force_col_wise": True
}

model = lgb.cv(
    params=config,
    train_set=dataset_cv_train,
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=10)
    ],
    nfold=5,
    return_cvbooster=True,
)

cvbooster = model["cvbooster"]

pred_per_cv = [booster.predict(test_features) for booster in cvbooster.boosters]
pred_per_cv = np.array(pred_per_cv)
print(pred_per_cv.shape)
pred_avg = pred_per_cv.mean(axis=0)
pred_per_cv = pred_per_cv.round(0)
print(pred_per_cv)

accs = []
pres = []
recs = []
f1s = []
for i in range(len(pred_per_cv)):
    accuracy = accuracy_score(test_labels, pred_per_cv[i])
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_per_cv[i], average="binary")
    accs.append(accuracy)
    pres.append(precision)
    recs.append(recall)
    f1s.append(f1)

# accs,pres,recs,f1sをresults/testにそれぞれcsvで保存
accs = pd.DataFrame([accs])
pres = pd.DataFrame([pres])
recs = pd.DataFrame([recs])
f1s = pd.DataFrame([f1s])

accs.to_csv("results/test/accs.csv", index=False, encoding="utf-8", header=False, mode="a", index_label=False)
pres.to_csv("results/test/pres.csv", index=False, encoding="utf-8", header=False, mode="a", index_label=False)
recs.to_csv("results/test/recs.csv", index=False, encoding="utf-8", header=False, mode="a", index_label=False)
f1s.to_csv("results/test/f1s.csv", index=False, encoding="utf-8", header=False, mode="a", index_label=False)

if "two" in option:
    pred_label = pred_avg.round(0)
    pred_label = pred_label.astype(int)
elif "three" in option:
    pred_label = np.argmax(pred_avg, axis=1)

for i in range(len(tokenized_test)):
    posts_token_ids = [id for id in tokenized_test["posts_tokenized"][i]]
    posts_decoded_tokens = [tokenizer.decode([token_id]) for token_id in posts_token_ids]
    posts_decoded_test_split.append(posts_decoded_tokens)

    posts_decoded_tokens_remove_special_tokens = []
    for token in posts_decoded_tokens:
        if "##" in token:
            token = token.replace("##", "")
        if "[CLS]" in token:
            token = token.replace("[CLS]", "")
        if "<s>" in token:
            token = token.replace("<s>", "")
        if "[UNK]" in token:
            token = token.replace("[UNK]", "")
        if "<unk>" in token:
            token = token.replace("<unk>", "")
        if "[SEP]" in token:
            token = token.replace("[SEP]", " ")
        if "</s>" in token:
            token = token.replace("</s>", " ")
        posts_decoded_tokens_remove_special_tokens.append(token)

    posts_decoded_test.append("".join(posts_decoded_tokens_remove_special_tokens))

    reposts_token_ids = [id for id in tokenized_test["reposts_tokenized"][i]]
    reposts_decoded_tokens = [tokenizer.decode([token_id]) for token_id in reposts_token_ids]
    reposts_decoded_test_split.append(reposts_decoded_tokens)

    reposts_decoded_tokens_remove_special_tokens = []
    for token in reposts_decoded_tokens:
        if "##" in token:
            token = token.replace("##", "")
        if "[CLS]" in token:
            token = token.replace("[CLS]", "")
        if "<s>" in token:
            token = token.replace("<s>", "")
        if "[UNK]" in token:
            token = token.replace("[UNK]", "")
        if "<unk>" in token:
            token = token.replace("<unk>", "")
        if "[SEP]" in token:
            token = token.replace("[SEP]", " ")
        if "</s>" in token:
            token = token.replace("</s>", " ")
        reposts_decoded_tokens_remove_special_tokens.append(token)

    reposts_decoded_test.append("".join(reposts_decoded_tokens_remove_special_tokens))

    if tokenized_test["flag_posts_tokenized"][i] is None:
        oc_decoded_tokens = []
    else:
        oc_token_ids = tokenized_test["flag_posts_tokenized"][i]
        oc_decoded_tokens = [tokenizer.decode(oc_token_ids[0])]

    flag_posts_decoded_test_split.append(oc_decoded_tokens)
    flag_posts_decoded_test.append(" ".join(oc_decoded_tokens))

tokenized_test["posts_decoded"] = posts_decoded_test
tokenized_test["reposts_decoded"] = reposts_decoded_test
tokenized_test["flag_posts_decoded"] = flag_posts_decoded_test

tokenized_test["pred"] = pred_label

# print(test_rates)
# print(pred_labels)

if not os.path.exists(f"results/Tweets/{model_name}"):
    os.makedirs(f"results/Tweets/{model_name}")

pickle.dump(cvbooster, open(f"results/Tweets/{model_name}/cvbooster.pkl", 'wb'))

# train_account_list = tokenized_train["account_name"]
# train_account_list.to_csv(f"results/{model_name}/train_account_list.csv", index=False, encoding="utf-8")

# valid_account_list = tokenized_valid["account_name"]
# valid_account_list.to_csv(f"results/{model_name}/valid_account_list.csv", index=False, encoding="utf-8")

# posts_tokenized_test.to_csv("results/test_sentences.csv", index=False, encoding="utf-8", columns=["label", "pred", "posts_decoded"])
if "two" in option:
    tp_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 1) & (tokenized_test["pred"] == 1)]
    fp_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 0) & (tokenized_test["pred"] == 1)]
    fn_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 1) & (tokenized_test["pred"] == 0)]
    tn_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 0) & (tokenized_test["pred"] == 0)]

    tp_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/tp_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "true_label", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    fp_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/fp_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "true_label", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    fn_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/fn_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "true_label", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    tn_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/tn_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "true_label", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])

    # posts_tokenized_test.to_csv("results/test_sentences.csv", index=False, encoding="utf-8", columns=["label", "pred", "posts_decoded"])

    cm = confusion_matrix(test_labels, pred_label)
    tn, fp, fn, tp = cm.flatten()
    accuracy = accuracy_score(test_labels, pred_label)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_label, average="binary")
    print(model_name)
    print("Confusion matrix")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print()
    # print(f"Predicted labels:\n{pred}")
    # print(f"True labels:\n{test_labels}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
elif "three" in option:
    t0_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 0) & (tokenized_test["pred"] == 0)]
    t1_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 1) & (tokenized_test["pred"] == 1)]
    t2_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 2) & (tokenized_test["pred"] == 2)]
    f0_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 0) & (tokenized_test["pred"] != 0)]
    f1_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 1) & (tokenized_test["pred"] != 1)]
    f2_post_tokenized_test = tokenized_test[(tokenized_test["label"] == 2) & (tokenized_test["pred"] != 2)]

    t0_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/t0_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    t1_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/t1_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    t2_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/t2_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    f0_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/f0_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    f1_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/f1_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])
    f2_post_tokenized_test.to_csv(f"results/Tweets/{model_name}/f2_sentences.csv", index=False, encoding="utf-8", columns=["account_name", "label", "pred", "flag_posts_decoded", "posts_decoded", "reposts_decoded"])

    cm = confusion_matrix(test_labels, pred_label, labels=[0, 1, 2])
    # tn, fp, fn, tp = cm.flatten()
    # confusion_matrixをプロットして保存
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["0", "1", "2"], rotation=45)
    plt.yticks(tick_marks, ["0", "1", "2"])
    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"results/Tweets/{model_name}/confusion_matrix.png")

    print(model_name)
    accuracy = balanced_accuracy_score(test_labels, pred_label)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_label, average="macro")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

raw_importance = cvbooster.feature_importance(importance_type="gain")
feature_names = cvbooster.boosters[0].feature_name()
cv_importance = pd.DataFrame(raw_importance, columns=feature_names)

sorted_importance = cv_importance.mean().sort_values(ascending=False)
df_sorted_importance = pd.DataFrame(sorted_importance, columns=["Importance"])

# それぞれのImportanceを合計値で割った値を計算し、新しい列に追加
df_sorted_importance["Importance_ratio"] = df_sorted_importance["Importance"] / df_sorted_importance["Importance"].sum()

print()
print(f"Posts embedding importance: {df_sorted_importance.filter(regex='^posts_embedding', axis=0)['Importance_ratio'].sum()}")
print(f"Reposts embedding importance: {df_sorted_importance.filter(regex='^reposts_embedding', axis=0)['Importance_ratio'].sum()}")
print(f"Posts Like importance: {df_sorted_importance.filter(regex='^posts_like', axis=0)['Importance_ratio'].sum()}")
print(f"Reposts Like importance: {df_sorted_importance.filter(regex='^reposts_like', axis=0)['Importance_ratio'].sum()}")
print(f"Posts Repost importance: {df_sorted_importance.filter(regex='^posts_repost', axis=0)['Importance_ratio'].sum()}")
print(f"Reposts Repost importance: {df_sorted_importance.filter(regex='^reposts_repost', axis=0)['Importance_ratio'].sum()}")

df_sorted_importance.to_csv(f"results/Tweets/{model_name}/feature_importance.csv", encoding="utf-8")
