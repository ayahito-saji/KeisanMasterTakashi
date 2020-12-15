import argparse
import os
import pickle

def transform(vocab, string):
    # 文字列をindexに変換するための辞書
    return [vocab[char] for char in string]

def transformer(vocab, dataset_path, dest_path):
    data = {
        "expressions": [],
        "answers": []
    }
    with open(dataset_path, "r") as fp:
        for line in fp.readlines():
        #line = fp.readline()
        #if True:
            expression, answer = line.rstrip().split("\t")
            data["expressions"].append(transform(vocab, expression))
            data["answers"].append(transform(vocab, '_'+answer))
    
    with open(dest_path, "wb") as fp:
        pickle.dump(data, fp)


def preprocess(dataset_dir_path, dest_dir_path, prefix):
    if prefix != "":
        train_file_name = f"{prefix}_train"
        test_file_name = f"{prefix}_test"
        vocab_file_name = f"{prefix}_vocab"
        revocab_file_name = f"{prefix}_revocab"
    else:
        train_file_name = "train"
        test_file_name = "test"
        vocab_file_name = "vocab"
        revocab_file_name = "revocab"

    vocab = {str(i): i for i in range(10)}
    vocab.update({" ":10, "+":11, "-":12, "*":13, "/":14, "(":15, ")":16, "_":17})

    revocab = {str(i): str(i) for i in range(10)}
    revocab.update({"10":"", "11":"+", "12":"-", "13":"*", "14":"/", "15":"(", "16":")", "17":""})


    print(20*"=", "Preprocessing Train Data", 20*"=")
    transformer(
        vocab,
        os.path.join(dataset_dir_path, train_file_name+".txt"),
        os.path.join(dest_dir_path, train_file_name+".pkl")
    )
    print(f"Preprocessed Train Data")

    print(20*"=", "Building Vocab Data", 20*"=")
    with open(os.path.join(dest_dir_path, vocab_file_name+".pkl"), "wb") as fp:
        pickle.dump(vocab, fp)
    print(f"Built Vocab Data")

    with open(os.path.join(dest_dir_path, revocab_file_name+".pkl"), "wb") as fp:
        pickle.dump(revocab, fp)
    print(f"Built Reverse Vocab Data")

    print(20*"=", "Preprocessing Test Data", 20*"=")
    transformer(
        vocab,
        os.path.join(dataset_dir_path, test_file_name+".txt"),
        os.path.join(dest_dir_path, test_file_name+".pkl")
    )
    print(f"Preprocessed Test Data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算数データセットの事前処理をします')
    parser.add_argument("--dataset_dir_path",
                        default="../data/datasets",
                        help="データセットのディレクトリのパス")

    parser.add_argument("--dest_dir_path",
                        default="../data/preprocessed",
                        help="前処理済データセットと語彙の保存先のディレクトリのパス")

    parser.add_argument("--prefix",
                        default="",
                        help="データセットの名前のprefixを指定")


    args = parser.parse_args()
    preprocess(
        dataset_dir_path=args.dataset_dir_path,
        dest_dir_path=args.dest_dir_path,
        prefix=args.prefix,
    )