import torch
import torch.nn as nn
import torch.optim as optim
import pprint
from classes import SelfAttention, Transformer, TransformerBlock

def get_data_and_vocab():
    # Define training data
    training_data = {
        "how are you": "i am fine <end>",
        "tell me about yourself": "i love travelling, coding<end>",
        "what is your name": "Adam <end>",
        "who is nice": "Adam <end>",
        "where is Adam": "at home <end>",
        "how is Adam": "i dont know <end>",
        "who are you": "your companion <end>"
    }

    # Extract input and target phrases
    data_words = [k for k, _ in training_data.items()]
    target_words = [v for _, v in training_data.items()]

    vocabulary_words = list(set([element.lower() for nestedlist in [x.split(" ") for x in data_words] for element in nestedlist] + [element.lower() for nestedlist in [x.split(" ") for x in target_words] for element in nestedlist]))
    vocabulary_words.remove("<end>")
    vocabulary_words.append("<end>")
    vocabulary_words.insert(0, "")

    # Create mappings from word to index and index to word
    word_to_ix = {vocabulary_words[k].lower(): k for k in range(len(vocabulary_words))}
    ix_to_word = {v: k for k, v in word_to_ix.items()}

    return training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word

def words_to_tensor(seq_batch, device=None):
    index_batch = []

    for seq in seq_batch:
        word_list = seq.lower().split(" ")
        indices = [word_to_ix[word] for word in word_list if word in word_to_ix]
        t = torch.tensor(indices)
        if device is not None:
            t = t.to(device)
        index_batch.append(t)

    return pad_tensors(index_batch)

# Function to convert a tensor of indices to a list of sequences of words
def tensor_to_words(tensor):
    index_batch = tensor.cpu().numpy().tolist()
    res = []
    for indices in index_batch:
        words = []
        for ix in indices:
            words.append(ix_to_word[ix].lower())
            if ix == word_to_ix["<end>"]:
                break
        res.append(" ".join(words))
    return res

# Function to pad a list of tensors to the same length
def pad_tensors(list_of_tensors):
    tensor_count = len(list_of_tensors) if not torch.is_tensor(list_of_tensors) else list_of_tensors.shape[0]
    max_dim = max(t.shape[0] for t in list_of_tensors)
    res = []
    for t in list_of_tensors:
        res_t = torch.zeros(max_dim, *t.shape[1:]).type(t.dtype).to(t.device)
        res_t[:t.shape[0]] = t
        res.append(res_t)

    # Concatenate tensors along a new dimension
    res = torch.cat(res)
    firstDim = len(list_of_tensors)
    secondDim = max_dim

    return res.reshape(firstDim, secondDim, *res.shape[1:])

def infer_recursive(model, input_vectors, max_output_token_count=10):
    model.eval()
    outputs = []

    # Loop over sequences in the batch
    for i in range(input_vectors.shape[0]):
        print(f"Infering sequence {i}")
        input_vector = input_vectors[i].reshape(1, input_vectors.shape[1])
        predicted_sequence = []
        wc = 0

        with torch.no_grad():
            while True:
                output = model(input_vector)
                predicted_index = output[0, :].argmax().item()
                predicted_sequence.append(predicted_index)
                if predicted_index == word_to_ix['<end>'] or wc > max_output_token_count:
                    break
                input_vector = torch.cat([input_vector, torch.tensor([[predicted_index]])], dim=1)
                wc += 1
        outputs.append(torch.tensor(predicted_sequence))
    outputs = pad_tensors(outputs)
    return outputs

def train_recursive(model, data, targets, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    batch_size, token_count, token_count_out = data.shape[0], data.shape[1], targets.shape[1]

    # Loop over sequences in the batch
    for b in range(batch_size):
        end_encountered = False
        cur_count = 0
        while not end_encountered:
            target_vector = torch.zeros(model.vocab_size).to(data.device)

            if cur_count != token_count_out:
                expected_next_token_idx = targets[b, cur_count]
                target_vector[expected_next_token_idx] = 1

            if cur_count > 0:
                model_input = data[b].reshape(token_count).to(data.device)
                part_of_output = targets[b, :cur_count].to(data.device)
                model_input = torch.cat((model_input, part_of_output))
            else:
                model_input = data[b]
            out = model(model_input.reshape(1, token_count + cur_count))
            loss = criterion(out, target_vector.reshape(out.shape))
            total_loss += loss
            cur_count += 1

            if cur_count > token_count_out:
                end_encountered = True

    # Backpropagate gradients and update model parameters
    total_loss.backward()
    optimizer.step()
    return total_loss.item() / batch_size

def example_training_and_inference():
    vocab_size = len(word_to_ix)
    embed_size = 512
    num_layers = 4
    heads = 3

    device = torch.device("cpu")
    model = Transformer(vocab_size, embed_size, num_layers, heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    data = words_to_tensor(data_words, device=device)
    targets = words_to_tensor(target_words, device=device)

    for epoch in range(50):
        avg_loss = train_recursive(model, data, targets, optimizer, criterion)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    input_vector = words_to_tensor(data_words, device=device)
    predicted_vector = infer_recursive(model, input_vector)
    predicted_words = tensor_to_words(predicted_vector)

    # Print training data and model output
    print("\n\n\n")
    print("Training Data:")
    pprint.pprint(training_data)
    print("\n\n")
    print("Model Inference:")
    result_data = {data_words[k]: predicted_words[k] for k in range(len(predicted_words))}
    pprint.pprint(result_data)

