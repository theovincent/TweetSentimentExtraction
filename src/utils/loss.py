def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def mean_jaccard(ground_truths, predicted_strings):
    avg = 0
    for i in range(len(predicted_strings)):
        avg += jaccard(predicted_strings[i], ground_truths[i])

    return avg
