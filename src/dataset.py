import json
import pandas as pd
import lightgbm as lgb


class Dataset:
    def __init__(self, oc_path: str, onc_path: str, random_path: str):
        with open(oc_path, "r") as f:
            oc_json = json.load(f)

        # Setting DataFrame of Opinion Changers
        self.df_oc = pd.DataFrame(oc_json)
        self.df_oc["label"] = self.df_oc["label"].astype(int)
        self.df_oc["true_label"] = self.df_oc["label"]
        self.df_oc["posts_tokenized"] = self.df_oc["posts_tokenized"].apply(lambda x: x[0])
        self.df_oc["reposts_tokenized"] = self.df_oc["reposts_tokenized"].apply(lambda x: x[0])
        self.df_oc = self.df_oc.sample(frac=1, random_state=42).reset_index(drop=True)

        with open(onc_path, "r") as f:
            onc_json = json.load(f)

        # Setting DataFrame of Opinion Non Changers
        self.df_onc = pd.DataFrame(onc_json)
        self.df_onc["label"] = self.df_onc["label"].astype(int)
        self.df_onc["true_label"] = self.df_onc["label"]
        self.df_onc["label"] = 0
        self.df_onc["label"] = self.df_onc["label"].astype(int)
        self.df_onc["posts_tokenized"] = self.df_onc["posts_tokenized"].apply(lambda x: x[0])
        self.df_onc["reposts_tokenized"] = self.df_onc["reposts_tokenized"].apply(lambda x: x[0])
        self.df_onc = self.df_onc.sample(frac=1, random_state=42).reset_index(drop=True)

        with open(random_path, "r") as f:
            random_json = json.load(f)

        # Setting DataFrame of Random Users
        self.df_random = pd.DataFrame(random_json)
        self.df_random["label"] = self.df_random["label"].astype(int)
        self.df_random["true_label"] = self.df_random["label"]
        self.df_random["posts_tokenized"] = self.df_random["posts_tokenized"].apply(lambda x: x[0])
        self.df_random["reposts_tokenized"] = self.df_random["reposts_tokenized"].apply(lambda x: x[0])
        self.df_random = self.df_random.sample(frac=1, random_state=42).reset_index(drop=True)

    def create(self) -> tuple[lgb.Dataset, lgb.Dataset, lgb.Dataset]:
        # Create Test Dataset
        cleaned_test_accounts = pd.concat([pd.read_csv("data/test_users/oc_test_accounts.csv"), pd.read_csv("data/test_users/onc_test_accounts.csv")], ignore_index=True)
        df_oc_test = self.df_oc[self.df_oc["account_name"].isin(cleaned_test_accounts["account_name"])]
        df_oc_test = df_oc_test[df_oc_test["label"] == 1]

        df_onc_test = self.df_onc[self.df_onc["account_name"].isin(cleaned_test_accounts["account_name"])]
        df_onc_test = df_onc_test.sample(frac=1, random_state=0).reset_index(drop=True)
        # df_onc_test = df_onc_test[:24]
        df_onc_test = df_onc_test[:len(df_oc_test) // 2]
        # df_onc_test = df_onc_test[df_onc_test["true_label" == 2]][:len(df_oc_test)]

        df_random_test = self.df_random.sample(frac=1, random_state=0).reset_index(drop=True)
        # df_random_test = df_random_test[:23]
        df_random_test = df_random_test[:len(df_oc_test) // 2]
        # df_random_test = df_random_test[:len(df_oc_test)]

        df_zero_oc = self.df_oc[self.df_oc["label"] == 3]
        df_zero_oc = df_zero_oc.sample(frac=1, random_state=0).reset_index(drop=True)
        df_zero_oc["label"] = 0
        df_oc_test_zero = df_zero_oc[:23]
        df_zero_oc = df_zero_oc[23:]
        # df_oc_test_zero = df_zero_oc[:len(df_oc_test)]
        # df_zero_oc = df_zero_oc[len(df_oc_test):]

        df_test = pd.concat([df_oc_test, df_onc_test, df_random_test], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        # df_test = pd.concat([df_oc_test, df_oc_test_zero], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        # df_test = pd.concat([df_oc_test, df_random_test], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)

        self.df_oc = self.df_oc[~self.df_oc["account_name"].isin(df_test["account_name"])]
        self.df_onc = self.df_onc[~self.df_onc["account_name"].isin(df_test["account_name"])]
        self.df_random = self.df_random[~self.df_random["account_name"].isin(df_test["account_name"])]

        df_one = self.df_oc[self.df_oc["label"] == 1]
        print(len(df_one))
        # df_zero_oc = df_zero_oc[:len(df_one) // 3]
        # df_zero_oc = df_zero_oc[:len(df_one)]
        # print(len(df_zero_oc))
        # df_zero_onc = self.df_onc.sample(frac=1, random_state=0).reset_index(drop=True)[:len(df_one) // 3]
        df_zero_onc = self.df_onc.sample(frac=1, random_state=0).reset_index(drop=True)[:len(df_one) // 2]
        # df_zero_onc = self.df_onc[self.df_oc["ture_label"] == 2].sample(frac=1, random_state=0).reset_index(drop=True)[:len(df_one)]
        # if len(df_zero_onc) < len(df_one):
        #     df_zero_onc = pd.concat([df_zero_onc, self.df_onc[self.df_onc[["ture_label"] == 0]][:len(df_one) - len(df_zero_onc)]], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        # print(len(df_zero_onc))
        df_zero_random = self.df_random.sample(frac=1, random_state=0).reset_index(drop=True)[:len(df_one) // 2]
        print(len(df_zero_random))
        df_zero = pd.concat([df_zero_onc, df_zero_random], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        # df_zero = pd.concat([df_zero_oc], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        # df_zero = pd.concat([df_zero_random], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)

        n_train = int(len(df_one) * 0.8)

        df_one_train = df_one[:n_train]
        df_zero_train = df_zero[:n_train]
        df_one_valid = df_one[n_train:]
        df_zero_valid = df_zero[n_train:]

        df_cv_train = pd.concat([df_one, df_zero], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        df_train = pd.concat([df_one_train, df_zero_train], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)
        df_valid = pd.concat([df_one_valid, df_zero_valid], ignore_index=True, join="inner").sample(frac=1, random_state=0).reset_index(drop=True)

        y_cv_train = df_cv_train["label"]
        y_train = df_train["label"]
        y_valid = df_valid["label"]
        y_test = df_test["label"]

        tokenized_cv_train = df_cv_train[["account_name", "label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"]]
        tokenized_train = df_train[["account_name", "label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"]]
        tokenized_valid = df_valid[["account_name", "label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"]]
        tokenized_test = df_test[["account_name", "label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"]]

        # , "posts_like", "reposts_like", "posts_repost", "reposts_repost"

        X_cv_train = df_cv_train.drop(columns=["label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"])
        X_train = df_train.drop(columns=["label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"])
        X_valid = df_valid.drop(columns=["label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"])
        X_test = df_test.drop(columns=["label", "true_label", "posts_tokenized", "flag_posts_tokenized", "reposts_tokenized"])

        if "account_name" in X_train.columns:
            X_cv_train = X_cv_train.drop(columns=["account_name"])
            X_train = X_train.drop(columns=["account_name"])
            X_valid = X_valid.drop(columns=["account_name"])
            X_test = X_test.drop(columns=["account_name"])

        # for i in range(768):
            # X_cv_train = X_cv_train.drop(columns=[f"posts_embedding_{i+1}"])
            # X_train = X_train.drop(columns=[f"posts_embedding_{i+1}"])
            # X_valid = X_valid.drop(columns=[f"posts_embedding_{i+1}"])
            # X_test = X_test.drop(columns=[f"posts_embedding_{i+1}"])
            # X_cv_train = X_cv_train.drop(columns=[f"reposts_embedding_{i+1}"])
            # X_train = X_train.drop(columns=[f"reposts_embedding_{i+1}"])
            # X_valid = X_valid.drop(columns=[f"reposts_embedding_{i+1}"])
            # X_test = X_test.drop(columns=[f"reposts_embedding_{i+1}"])

        dataset_cv_train = lgb.Dataset(X_cv_train, y_cv_train)
        dataset_train = lgb.Dataset(X_train, y_train)
        dataset_valid = lgb.Dataset(X_valid, y_valid, reference=dataset_train)

        print("CV Train:", len(df_cv_train))
        print("Train:", len(df_train))
        print("Valid:", len(df_valid))
        print("Test:", len(df_test))

        return dataset_cv_train, tokenized_cv_train, dataset_train, tokenized_train, dataset_valid, tokenized_valid, X_test, y_test, tokenized_test
