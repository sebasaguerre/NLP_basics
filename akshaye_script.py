import matplotlib.pyplot as plt
def count_frequencies(self) -> None:
    """
    Count the frequencies of each n-gram.
    Fill in the code to count n-gram occurrences.
    """

    ngram_freq = dict()
    for ngram in self.ngrams:
        ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
    # sorted_ngrams = sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True)
    self.ngrams = ngram_freq

def calculate_probabilities(self) -> None:
    """
    Calculate probabilities of each n-gram based on its frequency. Add alpha smoothing separately.
    """
    count = {}
    for ngram, freq in self.ngrams.items():
        context = tuple(ngram[:-1])
        count[context] = count.get(context, 0) + freq

    vocab = set(word for ngram in self.ngrams for word in ngram)
    V = len(vocab)

    self.probabilities = {}
    for ngram, freq in self.ngrams.items():
        context = tuple(ngram[:-1])
        numerator = freq + self.alpha
        denominator = count[context] + self.alpha * V
        self.probabilities[ngram] = numerator / denominator
    

def most_frequent_ngrams(self, top_n: int = 10) -> list:
    """
    Return the most frequent n-grams and their probabilities.
    """

    if not self.probabilities:
        self.calculate_probabilities()

    sorted_probs = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
    sorted_grams =  sorted_probs[:top_n]

    return sorted_grams


def plot_embeddings(M_reduced, vocab):
    plt.figure(figsize=(10, 10))
    plt.scatter(M_reduced[:, 0], M_reduced[:, 1])
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(M_reduced[i, 0], M_reduced[i, 1]), fontsize=8)
    plt.title('Word Embeddings')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.savefig('./results/word_embeddings.png')
    plt.show()

words = ['movie', 'book', 'mysterious', 'story', 'fascinating', 'good', 'interesting', 'large', 'massive', 'huge']

M_reduced = reduce_to_k_dim(co_matrix, k=2)
plot_embeddings(M_reduced, vocab)