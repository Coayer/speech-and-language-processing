import math
import string
import os

"""
Implements naive Bayes text classification to determine which class an unseen document belongs to.
"""


def train(documents, classes):
    class_probabilities = {}
    word_probabilities = {}

    N_doc = sum([len(class_docs) for class_docs in documents])
    vocab = set([word for class_docs in documents for doc in class_docs for word in doc])

    for c in range(len(classes)):
        N_c = len(documents[c])
        class_probabilities[classes[c]] = math.log(N_c / N_doc)

        combined_doc = [word for doc in documents[c] for word in doc]
        print(combined_doc)
        for word in vocab:
            # +1 is Laplace smoothing, prevents assigning 0 probability to word not seen in current class but seen in
            # another log(0) is undefined, so it must be avoided
            word_probabilities[(word, classes[c])] = math.log((combined_doc.count(word) + 1) / sum([(combined_doc.count(w) + 1) for w in vocab]))

    return word_probabilities, class_probabilities, vocab


def test(test_doc_path, classes, vocab, class_probabilities, word_probabilities):
    test_doc_path = [word for word in tokenizer(test_doc_path) if word in vocab]    # removes unseen words from test doc

    results = []
    for c in range(len(classes)):
        results.append(sum([word_probabilities[(w, classes[c])] for w in test_doc_path]) + class_probabilities[classes[c]])

    max_val = max(results)
    max_at = results.index(max_val)
    return classes[max_at]


def tokenizer(path):
    with open(path, "r") as file:
        data = file.readlines()
    return normalizer(data)


def normalizer(data):
    formatted = []
    for i in range(2, len(data), 2):
        formatted.extend(data[i][:-2].translate(str.maketrans('', '', string.punctuation)).lower().split(" "))

    return formatted


politics = [tokenizer("politics/" + document) for document in os.listdir("politics")]
tech = [tokenizer("tech/" + document) for document in os.listdir("tech")]
classes = ["politics", "tech"]

word_probabilities, class_probabilities, vocab = train([politics, tech], classes)
print(test("006.txt", classes, vocab, class_probabilities, word_probabilities))
