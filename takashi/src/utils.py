import torch
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_loader(preprocessed_data_path, padding_idx):
    with open(preprocessed_data_path, "rb") as fp:
        data = pickle.load(fp)
    
    max_expression_length = max([len(expression) for expression in data["expressions"]])
    max_answer_length = max([len(answer) for answer in data["answers"]])

    expressions = torch.ones((len(data["expressions"]), max_expression_length + 1), dtype=torch.long, device=device) * padding_idx
    answers = torch.ones((len(data["answers"]), max_answer_length + 1), dtype=torch.long, device=device) * padding_idx

    for i in range(0, len(data["expressions"])):
        expressions[i][:len(data["expressions"][i])] = torch.tensor(data["expressions"][i])
        answers[i][:len(data["answers"][i])] = torch.tensor(data["answers"][i])
    return expressions, answers

def batching(items, batch_size):
    item_batches = []
    for begin_index in range(0, len(items), batch_size):
        end_index = begin_index + batch_size
        item_batches.append(items[begin_index:end_index])
    
    return item_batches