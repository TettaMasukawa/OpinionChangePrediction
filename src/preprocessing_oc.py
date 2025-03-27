import emoji
from glob import glob
import json
import numpy as np
from os import environ
import pandas as pd
import re
from tqdm import tqdm

from embedding import Embedding


environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU ID
np.random.seed(42)


def preprocess_text(text) -> str:
    word_list = [
        "自民", "維新", "公明", "共産", "民主", "教育", "知事", "立憲", "違憲", "国民", "ハラスメント", "セクハラ", "パワハラ",
        "れいわ", "社民", "参政", "大臣", "五輪", "オリンピック", "災害", "震災", "津波", "台風", "同性", "LGBT", "国家",
        "内閣", "移民", "在日", "南海トラフ", "憲法", "税", "原子力", "原発", "市長", "エネルギー", "マニフェスト", "ファシズム",
        "COVID-19", "コロナ", "ワクチン", "政権", "選挙", "トランプ", "バイデン", "政府", "岸田", "安倍", "ウイルス", "エイズ",
        "菅", "石破", "野田", "麻生", "鳩山", "少子化", "高齢", "労働", "雇用", "年金", "保険", "保障", "福祉", "経済",
        "デフレ", "インフレ", "防衛", "外交", "自衛", "国連", "日米", "日中", "日韓", "温暖化", "安楽死", "暗殺", "自殺",
        "再生可能", "持続可能", "SDGs", "防災", "日銀", "金融", "日本銀行", "財務省", "無償化", "戦争", "給食", "賃上げ",
        "外務省", "首相", "野党", "与党", "政治", "マイナンバー", "デジタル", "財政", "国防", "医療", "株", "軍", "記者",
        "脱炭素", "気候変動", "安保", "国際連合", "紛争", "グローバル化", "国際社会", "病院", "医師", "不信任", "いじめ",
        "法律", "裁判", "中国", "台湾", "香港", "北朝鮮", "ロシア", "ウクライナ", "イスラエル", "パレスチナ", "アフガニスタン",
        "イラク", "イラン", "シリア", "アフリカ", "中東", "アジア", "ヨーロッパ", "アメリカ", "政党", "政策", "議員", "議会",
        "朝日", "尖閣", "読売", "産経", "日経", "新聞", "報道", "メディア", "小泉", "陰謀", "フェイクニュース", "デマ",
        "天皇", "皇室", "宮内庁", "皇后", "創価", "宗教", "仏教", "キリスト", "イスラム", "教会", "裏金", "汚職", "監査",
        "無党派", "市民", "デモ", "抗議", "暴動", "テロ", "犯罪", "警察", "検察", "刑務所", "刑法", "刑事", "死刑", "竹島",
        "領土", "領海", "領空", "国境", "国土", "国歌", "大戦", "戦後", "戦犯", "戦没者", "賃金", "給料", "給与", "ストライキ",
        "火山", "米国", "英国", "フランス", "ドイツ", "イタリア", "スペイン", "ポルトガル", "オーストラリア", "カナダ", "ブラジル",
        "支持率", "信条", "信仰", "詐欺", "土砂", "豪雨", "洪水", "個人情報保護", "法案", "議席", "献金", "寄付", "パラリンピック",
        "違法", "違反", "タバコ", "酒", "麻薬", "覚せい剤", "大麻", "コカイン", "アルコール", "喫煙", "たばこ", "煙草",
        "衆議院", "参議院", "右翼", "左翼", "極右", "極左", "過激派", "独裁", "中道", "保守", "リベラル", "革新", "改革",
        "改憲", "世論", "ごぼう", "NHK", "所得", "貧困", "格差", "差別", "制裁", "インフラ", "発電", "電力", "電気",
        "東電", "東京電力", "福島", "ロックダウン", "自粛", "感染", "投票", "失言", "辞任", "閣僚", "手取り", "独立",
        "ジェンダー", "人権", "条例", "仮想通貨", "ビットコイン", "失業", "解雇", "リストラ", "性別", "性差", "暴力",
        "公約", "公務", "公共", "反社", "ヤクザ", "交通", "患者", "NPO", "ボランティア", "オウム", "カルト", "池田大作",
        "統一教会", "幸福の科学", "別姓", "ベーシックインカム", "DX", "AI", "人工知能", "プロパガンダ", "オバマ", "クリントン",
        "黒人", "白人", "ヒスパニック", "ラテン", "ユダヤ", "クルド", "アラブ", "パンデミック", "公害", "環境", "ワーキングホリデー",
        "ワーホリ", "出稼ぎ", "留学", "核", "天下り", "兵役", "徴兵", "兵器", "情勢", "大統領", "首脳", "GDP", "不祥事",
        "世界遺産", "WHO", "TPP", "都構想", "利権", "保育", "介護", "障害", "障碍", "反ワク", "反日", "嫌韓", "森友", "加計",
        "反戦", "マイノリティ", "支持者", "工作員", "スパイ", "諜報", "物価", "貿易", "輸出", "輸入", "入国", "国籍", "指名手配",
        "逮捕", "文部", "DV", "インフル", "ＮＨＫ", "ＮＰＯ", "ＤＸ", "ＡＩ", "ＧＤＰ", "ＷＨＯ", "ＴＰＰ", "ＤＶ", "ＬＧＢＴ",
        "地下鉄", "基地", "オスプレイ", "原爆", "被曝", "放射能", "円高", "円安", "難民", "移住", "出生", "出産", "妊娠",
        "結婚", "ごみ", "リサイクル", "汚染", "ゴミ", "非正規", "立法", "離職", "キャッシュレス", "賭博", "ギャンブル",
        "赤字", "黒字", "為替", "地方創生", "ネトウヨ", "農業", "漁業", "産業", "復興", "ヘイトスピーチ", "メガバンク",
        "成人", "未成年", "民法", "判決", "冤罪", "大企業", "ベンチャー", "無所属", "国債", "都民", "県民", "民衆", "自死",
        "独居", "孤独死", "子育て", "騒音", "終戦", "教祖", "教団", "教科書", "補償", "石油", "ガソリン", "電子マネー",
        "高騰", "終活", "看護", "NISA", "資金", "金利", "融資", "インボイス", "事故", "みずほ", "三菱", "三井", "文書",
        "証券", "郵政", "民間", "賭け", "サイバー", "殺人", "暴行", "不動産", "団塊", "ゆとり", "生活習慣", "残業", "電通",
        "著作権", "特許", "可決", "否決", "不法", "解散", "任期", "モラハラ", "予算", "石丸", "小池百合子", "小沢", "離党",
        "プーチン", "主義", "資本", "ナチス", "男尊女卑", "人道", "平等", "多様性", "文化", "倫理", "道徳", "演説", "脅迫",
        "糾弾"
    ]

    for word in word_list:
        if word in text:
            text = text.replace(word, "")

    # Remove Mentions
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)

    # Remove URLs
    text = re.sub(r"https?://[w/:%#$&?()~.=+-…]+", "", text)

    # Remove Hashtags
    text = re.sub(r"#(\w+)", "", text)

    # Remove RT
    text = re.sub(r"RT ", "", text)

    # Remove Emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove 2 or more EOS
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()

    # &amp;, &lt;, &gt;, &quot;, &nbsp;
    for bf, af in zip(["&amp;", "&lt;", "&gt;", "&quot;", "&nbsp;"], ["&", "<", ">", "\"", " "]):
        text = text.replace(bf, af)

    # Remove \u3000
    text = text.replace("\u3000", " ")

    return text


oc_word_list = [
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
    "立場変わ", "見方変え", "見方変わ", "見解変え", "見解変わ", "考え改め", "もう支持", "もう賛同", "もう賛成", "もう同意", "支持してた",
    "支持してました", "支持していた", "支持していました", "賛成してた", "賛成してました", "賛成していた", "賛成していました", "賛同してた",
    "賛同してました", "賛同していた", "賛同していました", "同意してた", "同意してました", "同意していた", "同意していました", "支持していない",
]

embedding_func = Embedding("tohoku-nlp/bert-base-japanese-v3")
# embedding_func = Embedding("/home/rtakasu/sotuken/pre_trained_/until_250_len/checkpoint-550000/")
# embedding_func = Embedding("MU-Kindai/Japanese-SimCSE-BERT-base-sup")
# embedding_func = Embedding("/home/anakada/style_sentence_vector/saved_model/twitter_sup_simcse_base_hard_negative_100per/")

users = sorted(glob("data/opinion_changers/users/*.csv"))

cnt = 0

opinion_changer_features = []

oc_post_embeddings = []
oc_repost_embeddings_dict = []

for user in tqdm(users):
    df = pd.read_csv(user, low_memory=False)

    if 1 in df["flag"].values:
        flag_idx = df[df["flag"] == 1].index.tolist()[-1]
        flag_df = df.iloc[flag_idx]

        if len(flag_df["Content"]) == 0:
            continue

        user_handle = flag_df["Handle"]

        df.drop(range(0, flag_idx + 1), inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        continue

    word_match = False

    for oc_word in oc_word_list:
        if oc_word in flag_df["Content"]:
            word_match = True
            break

    if not word_match:
        continue

    df["Likes"] = df["Likes"].apply(lambda x: int(float(x[:-1]) * 1000) if "K" in str(x) else int(float(x[:-1]) * 1000000) if "M" in str(x) else int(x))
    df["Retweets"] = df["Retweets"].apply(lambda x: int(float(x[:-1]) * 1000) if "K" in str(x) else int(float(x[:-1]) * 1000000) if "M" in str(x) else int(x))
    df["Content"] = df["Content"].apply(preprocess_text)

    ex_word_list = [
        "プレゼント", "キャンペーン", "割引", "お得", "セール", "特典", "クーポン", "福袋", "キャッシュバック", "ポイント",
        "応募", "当選", "抽選", "英会話", "特設サイト"
    ]

    ex_word_idx = df[df["Content"].str.contains("|".join(ex_word_list))].index.tolist()
    df.drop(ex_word_idx, inplace=True)
    df.reset_index(drop=True, inplace=True)

    posts_raw = []
    reposts_raw = []

    for i in range(len(df)):
        if df["Handle"][i] == user_handle:
            posts_raw.append(df.iloc[i].to_dict())
        else:
            reposts_raw.append(df.iloc[i].to_dict())

    if len(posts_raw) < 1 or len(reposts_raw) < 1:
        continue

    posts_list = [posts_raw[i:i + 20] for i in range(0, len(posts_raw), 20)]
    repost_list = [reposts_raw[i:i + 20] for i in range(0, len(reposts_raw), 20)]
    reposts = []

    for i in range(len(posts_list)):
        if i > 2:
            break
        if i <= len(repost_list) - 1:
            reposts = repost_list[i]

        posts_like = sum([post["Likes"] for post in posts_list[i]]) if len(posts_list[i]) != 0 else 0
        reposts_like = sum([repost["Likes"] for repost in reposts]) if len(reposts) != 0 else 0
        posts_repost = sum([post["Retweets"] for post in posts_list[i]]) if len(posts_list[i]) != 0 else 0
        reposts_repost = sum([repost["Retweets"] for repost in reposts]) if len(reposts) != 0 else 0

        flag_posts_tokenized = embedding_func.get_tokens(flag_df["Content"])

        # Embeddings by Random
        posts_embeddings, posts_tokenized, num_added_posts = embedding_func.get_embeddings_by_random([post["Content"] for post in posts_list[i]])
        reposts_embeddings, reposts_tokenized, num_added_reposts = embedding_func.get_embeddings_by_random([repost["Content"] for repost in reposts])

        # Embeddings by Sorted
        # posts_embeddings, posts_tokenized, num_added_posts = embedding_func.get_embeddings_by_sorted([post["Content"] for post in posts_list[i]])
        # reposts_embeddings, reposts_tokenized, num_added_reposts = embedding_func.get_embeddings_by_sorted([repost["Content"] for repost in reposts])

        features = {
            "account_name": user_handle.replace("@", ""),
            "posts_like": posts_like,
            "reposts_like": reposts_like,
            "posts_repost": posts_repost,
            "reposts_repost": reposts_repost,
            "label": 1 if i == 0 else 3,
            "posts_tokenized": posts_tokenized,
            "reposts_tokenized": reposts_tokenized,
            "flag_posts_tokenized": flag_posts_tokenized
        }

        for n in range(len(posts_embeddings)):
            features[f"posts_embedding_{n+1}"] = posts_embeddings[n]
            features[f"reposts_embedding_{n+1}"] = reposts_embeddings[n]

        opinion_changer_features.append(features)

    cnt += 1

# Set the path to save the features
with open("data/features/oc_features_tweets_random-BERT.json", "w") as f:
    json.dump(opinion_changer_features, f, indent=4, ensure_ascii=False)
