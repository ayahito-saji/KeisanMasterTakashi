import argparse
import os
import pickle
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Encoder, Decoder
from utils import data_loader, batching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(preprocessed_dir_path,
         dest_dir_path,
         prefix,
         embedding_dim,
         hidden_dim,
         epoch,
         learning_rate,
         batch_size):
    print(20*"=", f"Preparing for train", 20*"=")

    if prefix != "":
        train_file_name = f"{prefix}_train"
        vocab_file_name = f"{prefix}_vocab"
        weight_file_name = f"{prefix}_weight"
    else:
        train_file_name = "train"
        vocab_file_name = "vocab"
        weight_file_name = "weight"

    with open(os.path.join(preprocessed_dir_path, vocab_file_name+".pkl"), "rb") as fp:
        vocab = pickle.load(fp)

    vocab_size = len(vocab)
    padding_idx = vocab[" "]
    
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, padding_idx).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, padding_idx).to(device)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    expressions, answers = data_loader(os.path.join(preprocessed_dir_path, train_file_name+".pkl"), padding_idx)

    expression_batches = batching(expressions, batch_size)
    answer_batches = batching(answers, batch_size)

    print(20*"=", f"Training", 20*"=")

    all_losses = []
    all_times = []

    for i in range(1, epoch+1):
        print(f"Epoch {i}")
        epoch_loss = 0
        start_time = time.time()

        for j in range(len(expression_batches)):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            expression_batch = expression_batches[j]
            answer_batch = answer_batches[j]

            encoder_state = encoder(expression_batch)

            source = answer_batch[:, :-1]
            target = answer_batch[:, 1:]

            loss = 0

            decoder_output, _ = decoder(source, encoder_state)

            for k in range(decoder_output.size()[1]):
                loss += criterion(decoder_output[:, k, :], target[:, k])        
            epoch_loss += loss.item()

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

        end_time = time.time()
        t = end_time - start_time
        print(f"  Loss:{epoch_loss} Time:{datetime.timedelta(seconds=int(t))}s")

        all_losses.append(epoch_loss)
        all_times.append(t)
        
        if min(all_losses) == epoch_loss:
            torch.save({ "encoder": encoder.state_dict(), "decoder": decoder.state_dict() }, os.path.join(dest_dir_path, f"{weight_file_name}.pth"))

        if epoch_loss < 1:
            print("  Ealry Stop")
            break

    print(20*"=", f"Done training", 20*"=")
    print(f"Minimum Loss: {min(all_losses)}")
    print(f"Training time: {datetime.timedelta(seconds=int(sum(all_times)))}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='たかしの学習をします')
    parser.add_argument("--preprocessed_dir_path",
                        default="../data/preprocessed",
                        help="前処理済みデータセットのディレクトリのパス")

    parser.add_argument("--dest_dir_path",
                        default="../data/trained",
                        help="学習済みたかしの保存先のディレクトリのパス")

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
    
    parser.add_argument("--epoch",
                        default=100,
                        type=int,
                        help="エポック数")

    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="学習率の設定")

    parser.add_argument("--batch_size",
                        default=100,
                        type=int,
                        help="バッチ数")


    args = parser.parse_args()

    main(
        preprocessed_dir_path=args.preprocessed_dir_path,
        dest_dir_path=args.dest_dir_path,
        prefix=args.prefix,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )