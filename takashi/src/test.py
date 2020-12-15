import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Encoder, Decoder
from utils import data_loader, batching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vocab_indexes(decoder_output, padding_idx):
    results = []
    all_padding_idx = True
    for h in decoder_output:
        idx = torch.argmax(h)
        if idx != torch.tensor(padding_idx):
            all_padding_idx = False
        results.append(idx)
    return torch.tensor(results, device=device).view(len(results), 1), all_padding_idx

def predict(encoder, decoder, vocab, revocab, expressions, batch_size, padding_idx):
    expression_batches = batching(expressions, batch_size)

    results = []
    for j in range(len(expression_batches)):
        with torch.no_grad():
            expression_batch = expression_batches[j]

            decoder_hidden = encoder(expression_batch)

            decoder_input = torch.tensor([[vocab["_"]] for _ in range(batch_size)], device=device)
            batch_results = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            for i in range(100):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                decoder_input, all_padding_idx = get_vocab_indexes(decoder_output.squeeze(), padding_idx)
                batch_results = torch.cat([batch_results, decoder_input], dim=1)
                if all_padding_idx == True:
                    break
            
            batch_results = batch_results[:,1:].detach().numpy()
            for ids in batch_results:
                result = int("".join([ revocab[str(id)] for id in ids]))    
                results.append(result)
    return results


def main(preprocessed_dir_path, trained_dir_path, prefix, embedding_dim, hidden_dim, batch_size):
    print(20*"=", f"Preparing for test", 20*"=")
    if prefix != "":
        test_file_name = f"{prefix}_test"
        vocab_file_name = f"{prefix}_vocab"
        revocab_file_name = f"{prefix}_revocab"
        weight_file_name = f"{prefix}_weight"
    else:
        test_file_name = "test"
        vocab_file_name = "vocab"
        revocab_file_name = "revocab"
        weight_file_name = "weight"

    with open(os.path.join(preprocessed_dir_path, vocab_file_name+".pkl"), "rb") as fp:
        vocab = pickle.load(fp)

    with open(os.path.join(preprocessed_dir_path, revocab_file_name+".pkl"), "rb") as fp:
        revocab = pickle.load(fp)


    vocab_size = len(vocab)
    padding_idx = vocab[" "]

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, padding_idx).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, padding_idx).to(device)
    encoder.load_state_dict(torch.load(os.path.join(trained_dir_path, f"{weight_file_name}.pth"))["encoder"])
    decoder.load_state_dict(torch.load(os.path.join(trained_dir_path, f"{weight_file_name}.pth"))["decoder"])

    expression_ids, answer_ids = data_loader(os.path.join(preprocessed_dir_path, test_file_name+".pkl"), padding_idx)

    print(20*"=", f"Testing", 20*"=")
    predicted_answers = predict(encoder, decoder, vocab, revocab, expression_ids, batch_size, padding_idx)

    answers = []
    for ids in answer_ids.detach().numpy().tolist():
        answer = int("".join([revocab[str(id)] for id in ids]))    
        answers.append(answer)
    
    print(20*"=", f"Calculate Test Result", 20*"=")
    score = 0
    missed = []
    for i, answer in enumerate(answers):
        if predicted_answers[i] == answer:
            score += 1
        else:
            missed.append((answer, predicted_answers[i]))

    print(f"Accuracy: {score/len(answers) * 100} ({score}/{len(answers)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='たかしのテストをします')
    parser.add_argument("--preprocessed_dir_path",
                        default="../data/preprocessed",
                        help="前処理済みデータセットのディレクトリのパス")

    parser.add_argument("--trained_dir_path",
                        default="../data/trained",
                        help="学習済みたかしのディレクトリのパス")

    parser.add_argument("-p", "--prefix",
                        default="",
                        help="データセットの名前のprefixを指定")

    parser.add_argument("--embedding_dim",
                        default=200,
                        type=int,
                        help="文字のEmbeddingの次元数")

    parser.add_argument("--hidden_dim",
                        default=128,
                        type=int,
                        help="LSTMの隠れ層の次元数")

    parser.add_argument("--batch_size",
                        default=100,
                        type=int,
                        help="バッチ数")


    args = parser.parse_args()

    main(
        preprocessed_dir_path=args.preprocessed_dir_path,
        trained_dir_path=args.trained_dir_path,
        prefix=args.prefix,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size
    )