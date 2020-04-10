import string
import os
import numpy as np


def pagerank(A, d=0.85, e=0.01):
    n = len(A)
    dM = d * np.transpose(np.dot(np.linalg.inv(np.diagflat([np.sum(A[i]) for i in range(n)])), A))
    damping = np.ones(n) * (1 - d) / n
    R = np.ones(n) / n
    prev = np.zeros(n)

    while np.sum(R - prev) > e:
        prev = R
        R = np.dot(dM, R) + damping

    return R


def remove_nonalpha(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation + "0123456789")).lower()


def make_words_vector(document):
    words = []
    for sentence in document:
        for word in remove_nonalpha(sentence).split(" "):
            if word not in words:
                words.append(word)

    return words


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))


def sentence2vec(sentence,words, document, corpus):
    sentence_vector = np.zeros(len(words))
    sentence = remove_nonalpha(sentence)

    for i in range(len(words)):
        if words[i] in sentence:
            sentence_vector[i] = tf_idf(words[i], document, corpus)

    return sentence_vector


def tf_idf(term, document, corpus):
    term_frequency = np.log(1 + np.sum([sentence.count(term) for sentence in document]))
    # np.sum([sentence.count(term) for sentence in document])
    inverse_document_frequency = np.log(
        len(corpus) / (1 + np.sum([in_doc(term, doc) for doc in corpus])))  # +1 prevents division by zero
    return term_frequency * inverse_document_frequency


def in_doc(term, document):
    for sentence in document:
        if term in remove_nonalpha(sentence).split(" "):
            return True

    return False


def read_doc(path):
    with open(path, "r") as file:
        data = file.readlines()

    formatted = []
    for i in range(2, len(data), 2):
        formatted.extend(data[i][:-2].split(". "))

    return formatted


def main():
    corpus = [read_doc("corpus/" + document) for document in os.listdir("corpus")]
    document = corpus[14]

    words = make_words_vector(document)
    sentence_vectors = [sentence2vec(sentence, words, document, corpus) for sentence in document]

    n = len(document)
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            adjacency_matrix[i, j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j])

    sentence_scores = pagerank(adjacency_matrix)
    mean = np.mean(sentence_scores)

    for i in range(len(sentence_scores)):
        if sentence_scores[i] > mean:
            print(document[i] + ".")


main()
