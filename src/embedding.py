import random
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import torch


class Embedding:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mecab_kwargs={"mecab_dic": None, "mecab_option": "-d {0}".format("/usr/local/lib/mecab/dic/mecab-ipadic-neologd")}
        )

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # Using JTweetRoBERTa

        self.model = AutoModel.from_pretrained(
            model_name,
            return_dict=True,
            output_hidden_states=True
        )

        self.model.eval()
        self.model.to("cuda")

        self.simcse_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mecab_kwargs={"mecab_dic": None, "mecab_option": "-d {0}".format("/usr/local/lib/mecab/dic/mecab-ipadic-neologd")}
        )

        # self.simcse_tokenizer = AutoTokenizer.from_pretrained(model_name)  # Using JTweetRoBERTa

        self.simcse_model = AutoModel.from_pretrained(
            model_name,
            return_dict=True,
            output_hidden_states=True
        )

        self.simcse_model.eval()
        self.simcse_model.to("cuda")

    def create_mask(self, input_ids, pad_token_id, sep_token_id):
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        for i, row in enumerate(input_ids):
            for j, token_id in enumerate(row):
                if token_id == pad_token_id or token_id == sep_token_id:
                    mask[i, j] = False

        return mask

    def get_tokens(self, text):
        tokenized_text = self.tokenizer.encode_plus(text, return_tensors="pt")
        return tokenized_text["input_ids"].tolist()

    def get_embeddings(self, text):
        tokenized_text = self.tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            all_encoder_layers = self.model(**tokenized_text)

        embeddings = all_encoder_layers["pooler_output"][0].cpu()

        return embeddings.tolist(), tokenized_text["input_ids"].cpu().tolist()

    def count_tokens(self, sentences):
        combined_tokens = []
        added_sentences = []

        for sentence in sentences:
            if sentence == "":
                continue
            else:
                sentence_tokens = self.tokenizer.tokenize(sentence) + ["[SEP]"]
                # sentence_tokens = self.tokenizer.tokenize(sentence) + ["</s>"]  # Using JTweetRoBERTa

            if len(combined_tokens) + len(sentence_tokens) <= 512:
            # if len(combined_tokens) + len(sentence_tokens) <= 250: # Using JTweetRoBERTa
                sentence_tokens = [token.replace("#", "") for token in sentence_tokens]

                combined_tokens.extend(sentence_tokens)
                added_sentences.append(sentence)
            else:
                break

        return combined_tokens, added_sentences

    def get_embeddings_by_random(self, sentences):
        random.seed(42)

        combined_tokens, added_sentences = self.count_tokens(sentences)

        random.shuffle(combined_tokens)
        random.shuffle(added_sentences)
        combined_tokens = "".join(combined_tokens)
        added_sentences = "[SEP]".join(added_sentences)
        # added_sentences = "</s>".join(added_sentences)  # Using JTweetRoBERTa

        tokenized_text = self.tokenizer.encode_plus(added_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            all_encoder_layers = self.model(**tokenized_text)

        embeddings = all_encoder_layers["pooler_output"][0].cpu()

        return embeddings.tolist(), tokenized_text["input_ids"].cpu().tolist(), len(added_sentences)

    def get_embeddings_by_sorted(self, sentences):
        combined_tokens, added_sentences = self.count_tokens(sentences)
        combined_tokens = "".join(combined_tokens)
        combined_tokens = "".join(combined_tokens)
        added_sentences = "[SEP]".join(added_sentences)
        # added_sentences = "</s>".join(added_sentences)  # Using JTweetRoBERTa

        tokenized_text = self.tokenizer.encode_plus(added_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            all_encoder_layers = self.model(**tokenized_text)

        embeddings = all_encoder_layers["pooler_output"][0].cpu()

        return embeddings.tolist(), tokenized_text["input_ids"].cpu().tolist(), len(added_sentences)

    def get_embeddings_by_topk(self, sentences, flag_sentence):
        word_list = [
            "意見を変えた", "考えを変えた", "支持を変えた", "立場を変えた", "見方を変えた", "見解を変えた", "意見を変えました",
            "考えを変えました", "支持を変えました", "立場を変えました", "見方を変えました", "見解を変えました", "考えを改めた", "考えを改めました",
            "もう支持しない", "もう賛成しない", "もう賛同しない", "もう同意しない", "もう支持できない", "もう賛成できない", "もう賛同できない",
            "もう同意できない", "もう支持したくない", "もう賛成したくない", "もう賛同したくない", "もう同意したくない", "もう支持しません",
            "もう賛成しません", "もう賛同しません", "もう同意しません", "もう支持できません", "もう賛成できません", "もう賛同できません",
            "もう同意できません", "もう支持したくありません", "もう賛成したくありません", "もう賛同したくありません", "もう同意したくありません",
            "支持やめた", "支持やめました", "支持しない", "賛成しない", "賛同しない", "同意しない", "支持できない", "賛成できない", "賛同できない",
            "同意できない", "支持したくない", "賛成したくない", "賛同したくない", "同意したくない", "支持しません", "賛成しません", "賛同しません",
            "同意しません", "支持できません", "賛成できません", "賛同できません", "同意できません", "支持したくありません", "賛成したくありません",
            "賛同したくありません", "同意したくありません", "意見変え", "意見変わ", "考え変え", "考え変わ", "支持変え", "支持変わ", "立場変え",
            "立場変わ", "見方変え", "見方変わ", "見解変え", "見解変わ", "考え改め", "もう支持", "もう賛同", "もう賛成", "もう同意",
            "支持", "賛成", "賛同", "同意", "意見", "考え", "立場", "見方", "見解"
        ]

        formatted_sentences = []

        for sentence in sentences:
            for word in word_list:
                if word in sentence:
                    sentence = sentence.replace(word, "")

            formatted_sentences.append(sentence)

        combined_tokens, added_sentences = self.count_tokens(formatted_sentences)

        flag_tokenized_text = self.simcse_tokenizer(flag_sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            flag_embeddings = self.simcse_model(**flag_tokenized_text).pooler_output[0].cpu()

        flag_embeddings = flag_embeddings.tolist()

        sentences_detaloader = DataLoader(added_sentences, batch_size=64, shuffle=False)
        distances_with_sentences = []

        for batch_sentences in sentences_detaloader:
            sentences_tokenized_text = self.simcse_tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

            with torch.no_grad():
                sentences_embeddings = self.simcse_model(**sentences_tokenized_text).pooler_output.cpu()

            for i in range(len(sentences_embeddings)):
                post_embeddings = sentences_embeddings[i].tolist()
                distance = cosine(flag_embeddings, post_embeddings)
                distances_with_sentences.append((distance, added_sentences[i]))

        sorted_distances_with_sentences = sorted(distances_with_sentences, key=lambda x: x[0])
        sorted_sentences = [sentence for _, sentence in sorted_distances_with_sentences]

        posts_embeddings, posts_tokenized, len_sentence = self.get_embeddings_by_sorted(sorted_sentences)

        return posts_embeddings, posts_tokenized, len_sentence

    def reduce_embeddings(self, embeddings: torch.Tensor):
        embeddings = self.umap.fit_transform(embeddings)

        return embeddings
