import torch
import torch.nn as nn
import torch.optim as optim
import pprint
from functions import get_data_and_vocab, words_to_tensor, tensor_to_words, pad_tensors, infer_recursive, train_recursive, example_training_and_inference

def main():
    training_data, data_words, target_words, vocabulary_words, word_to_ix, ix_to_word = get_data_and_vocab()

    example_training_and_inference()


if __name__=="__main__":
    main()