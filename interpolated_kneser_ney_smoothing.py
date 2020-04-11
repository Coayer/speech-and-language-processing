import numpy as np

"""
Implements an interpolated Kneser-Ney smoothing algorithm using a bigram model.

Simpler methods of smoothing, like Laplace (AKA add-1, increases count by one for all words) can be used to prevent
a language model from assigning a probability of 0 to known words in unknown contexts. Kneser-Ney smoothing is a form 
of interpolated smoothing which assigns higher probabilities to words of low counts which appear in a higher number 
of bigrams, and reduces the probability of frequently used words which occur in few contexts.

This implementation of the algorithm uses only bigram modelling, though Kneser-Ney smoothing supports N-grams.

I chose not to use any unknown word handling as it would introduce further complexity without being very interesting 
to code. This means input sentences can only contain previously-seen words.
"""


def interpolated_kneser_ney(previous, current):
    global d, tokens
    bigram = np.maximum(bigram_count(previous, current) - d, 0) / tokens.count(previous)
    unigram = normalizer(previous) * p_cont(current)
    return bigram + unigram


def bigram_count(w1, w2):
    global tokens
    total = 0

    for i in np.arange(len(tokens) - 1):
        if tokens[i] == w1 and tokens[i + 1] == w2:
            total += 1

    return total


def normalizer(w):
    global d, tokens
    normalized_discount = d / np.sum([bigram_count(w, v) for v in tokens])
    times_discount_applied = len([1 for v in tokens if (bigram_count(w, v) > 0)])
    return normalized_discount * times_discount_applied


def p_cont(w):
    global tokens, unique_bigrams
    novel_continuations = len(set([(v, w) for v in tokens if bigram_count(v, w) > 0]))
    return novel_continuations / unique_bigrams


def tokenizer(path):
    with open(path, "r") as file:
        data = file.readlines()

    formatted = []
    import string
    for i in range(2, len(data), 2):
        formatted.extend(data[i][:-2].translate(str.maketrans('', '', string.punctuation)).lower().split(" "))

    return formatted


file = "politics/007.txt"
tokens = tokenizer(file)
print(tokens)

d = 0.75
unique_bigrams = len(set([(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]))

while True:
    sentence = input("Enter sentence:\n").split(" ")
    bigram_probabilities = [interpolated_kneser_ney(sentence[i], sentence[i + 1]) for i in np.arange(len(sentence) - 1)]
    total_probability = np.exp(np.sum([np.log(p) for p in bigram_probabilities]))

    print(bigram_probabilities)
    print("Probability: " + str(total_probability))
    print("Perplexity: " + str(total_probability ** (-1 / len(sentence))))
    print("\n")
