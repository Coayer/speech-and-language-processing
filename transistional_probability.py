import string
import pandas as pd
import numpy as np

letters = list(" " + string.ascii_lowercase)

transitional_prob = pd.DataFrame(np.zeros((27, 27), dtype=int), columns=letters, index=letters)
letter_counts = dict(zip(letters, [0 for x in range(27)]))

with open("words.txt") as word_file:
    words = word_file.read().splitlines()

words = np.asarray(words)

for word in words:
    word = " " + word + " "
    for i in range(len(word) - 1):
        transitional_prob.at[word[i], word[i + 1]] += 1
        letter_counts[word[i]] += 1

print(transitional_prob.to_string())

for letter in letter_counts:
    print("{} : {}".format(letter, letter_counts[letter]))

while True:
    try:
        threshold = float(input("Enter threshold value:\n"))
        sentence = input("Enter speech:\n")

        result = ""

        for i in range(len(sentence) - 1):
            if (transitional_prob.at[sentence[i], sentence[i + 1]] / letter_counts[sentence[i]]) < threshold:
                result += sentence[i] + " "
            else:
                result += sentence[i]

        result += sentence[-1]

        print(result)
        
    except ValueError:
        continue
