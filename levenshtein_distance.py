# calculates distance between two strings using Levenshtein distances (minimum edit distance)

def minimum_edit_distance(source, target):
    # costs could be dependent on letter, as some letters are more likely to be inserted
    delete_cost = insert_cost = 1

    n = len(source)
    m = len(target)

    D = [[0 for j in range(m + 1)] for i in range(n + 1)]

    # empty string to empty string has 0 cost, so start at i = j = 1
    # all (+1)s in range() below (and -1 in substitution_cost call) due to Python range function not inclusive for end
    for i in range(1, n + 1):
        D[i][0] = D[i-1][0] + delete_cost
    for j in range(1, m + 1):
        D[0][j] = D[0][j-1] + insert_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min([D[i-1][j] + delete_cost,
                           D[i][j-1] + insert_cost,
                           D[i-1][j-1] + substitution_cost(source[i-1], target[j-1])])

    [print(row) for row in D]
    return D[n][m]


def substitution_cost(source_letter, target_letter):
    if source_letter != target_letter:
        return 2    # counts as doing delete and insert operation (1 + 1)
    else:
        return 0    # no change in letter, so no cost


print(minimum_edit_distance("intention", "execution"))
