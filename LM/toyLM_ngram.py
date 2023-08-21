###################
# INPUTS
###################

n = 3
max_words_gen = 30
max_words_sample = 50
number_to_generate = 3
smoothing_constant = 1

dataset_base = 'HerMajestySpeechesDataset'
#dataset_base = 'Hobbit'
#dataset_base = 'LOTR'
train_set_file = dataset_base + '/train.txt'
dev_set_file = dataset_base + '/dev.txt'
test_set_file = dataset_base + '/test.txt'

###################
# IMPORTS
###################

import random
from collections import Counter
from statistics import mean
from tensorflow.keras.preprocessing.text import Tokenizer


###################
# CLASSES
###################

class NGramModel:

    def __init__(self, n):
        self.n = n
        self.n_minus_1_grams = {}
        self.grams = set()
        self.tokenizer = Tokenizer(oov_token="<unk>")
        self.sos_token = -1
    
    def gen_start(self, n, start):
        return [start for i in range(n, 0, -1)]

    def get_ngrams(self, sequence, n, start, end):
        seq = self.gen_start(n-1, start) + sequence + [end]
        ngrams = [tuple(seq[i:i+n]) for i in range(0,len(seq)-1)]
        return ngrams

    def train(self, dataset):
        self.tokenizer.fit_on_texts(dataset)
        vocab_size = len(self.tokenizer.word_index) + 1 
        self.eos_token = len(self.tokenizer.word_index) + 1
        sequences = self.tokenizer.texts_to_sequences(dataset)
        for text in sequences:
            self.learn(text)

    def learn(self, sentence):
        if len(sentence) == 0:
            return
        ngrams = self.get_ngrams(sentence, self.n, self.sos_token, self.eos_token)
        for ngram in ngrams:
            nm1_gram = tuple(ngram[0:self.n-1])
            last_gram = ngram[-1]
            self.grams.add(tuple([last_gram]))
            nm1_gram_info = self.n_minus_1_grams.get(nm1_gram, {})
            last_gram_count = nm1_gram_info.get(last_gram, 0)
            nm1_gram_info[last_gram] = last_gram_count + 1
            self.n_minus_1_grams[nm1_gram] = nm1_gram_info

    def probability(self, context, word, laplacian_k=1):
        nm1_gram_info = self.n_minus_1_grams.get(context, {})
        total = sum(nm1_gram_info.values())
        vocabulary_size = len(self.grams)
        last_gram_count = nm1_gram_info.get(word, 0)
        if smoothing_constant == 0 and total == 0:
            return 1/float(vocabulary_size)
        return (last_gram_count + laplacian_k) / (float(total) + vocabulary_size * laplacian_k)

    def get_most_likely(self, context):
        nm1_gram_info = self.n_minus_1_grams.get(context, {})
        if not nm1_gram_info:
            return random.sample(self.grams, 1)
        max_word = max(nm1_gram_info, key=nm1_gram_info.get)
        return max_word

    def get_sampled(self, context, max_sample_words=None):
        nm1_gram_info = self.n_minus_1_grams.get(context, {})
        if not nm1_gram_info:
            return random.sample(self.grams, 1)
        keys = list(nm1_gram_info.keys())
        values = list(nm1_gram_info.values())
        if max_sample_words is not None:
            sorted_pairs = sorted(zip(values, keys), reverse=True)
            sorted_pairs = sorted_pairs[:max_sample_words]
            values, keys = zip(*sorted_pairs)
        samples = random.choices(keys, weights=values, k=1)
        return samples[0]

    def generate_most_likely(self, n):
        context = tuple(self.gen_start(self.n-1, self.sos_token))
        gen = []
        for i in range(n):
            next = self.get_most_likely(context)
            if next == self.eos_token:
                break
            gen.append(next)
            context = context[1:] + tuple([next])
        generated = self.tokenizer.sequences_to_texts([gen])[0]
        return generated

    def generate_sampling(self, n, max_sample_words=None):
        context = tuple(self.gen_start(self.n-1, self.sos_token))
        gen = []
        for i in range(n):
            next = self.get_sampled(context, max_sample_words)
            if next == self.eos_token:
                break
            gen.append(next)
            context = context[1:] + tuple([next])
        generated = self.tokenizer.sequences_to_texts([gen])[0]
        return generated

    def perplexity_text(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        return self.perplexity(sequence)

    def perplexity(self, sequence):
        ngrams = self.get_ngrams(sequence, self.n, self.sos_token, self.eos_token)
        totalp = 1.0
        n = len(sequence)
        for ngram in ngrams:
            nm1_gram = tuple(ngram[0:self.n-1])
            last_gram = ngram[-1]
            p = self.probability(nm1_gram, last_gram, smoothing_constant)
            if p > 0:
                totalp = p * totalp
        if totalp == 0:
            return float('inf')
        return (1/totalp)**(1/n)

    def perplexity_one(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        if len(sequence) == 0:
            return 0.0
        return self.perplexity(sequence)

    def perplexity_all(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = [s for s in sequences if len(s) > 0]
        ps = [self.perplexity(s) for s in sequences]
        return mean(ps)


###################
# MODEL
###################

print("")
print(f"{n}-GRAM MODEL")
nGramModel = NGramModel(n)


###################
# TRAIN
###################

file_train = open(train_set_file, 'r', encoding="utf8")
lines_train = file_train.readlines()
nGramModel.train(lines_train)


###################
# EVALUATE
###################

print("\nPERPLEXITY DEV")
file_dev = open(dev_set_file, 'r', encoding="utf8")
lines_dev = file_dev.readlines()
perplexity_dev = nGramModel.perplexity_all(lines_dev)
print('{0:.2f}'.format(perplexity_dev))

print("\nPERPLEXITY TEST")
file_test = open(test_set_file, 'r', encoding="utf8")
lines_test = file_test.readlines()
perplexity_test = nGramModel.perplexity_all(lines_test)
print('{0:.2f}'.format(perplexity_test))

for i in range(2):
    pp = '{0:.2f}'.format(nGramModel.perplexity_one(lines_dev[i]))
    print(f'\nPERPLEXITY = {pp}: "{lines_dev[i]}"')


###################
# GENERATE
###################

print(f"\nGENERATE MAX {max_words_gen} WORDS / MOST LIKELY")
generated = nGramModel.generate_most_likely(max_words_gen)
perplexity = '{0:.2f}'.format(nGramModel.perplexity_one(generated))
print(f"{1} (pp={perplexity}): {generated}")

print(f"\nGENERATE MAX {max_words_gen} WORDS / SAMPLING")
for i in range(number_to_generate):
    generated = nGramModel.generate_sampling(max_words_gen)
    perplexity = '{0:.2f}'.format(nGramModel.perplexity_one(generated))
    print(f"{i+1} (pp={perplexity}): {generated}")

print(f"\nGENERATE MAX {max_words_gen} WORDS / SAMPLING {max_words_gen} MOST LIKELY")
for i in range(number_to_generate):
    generated = nGramModel.generate_sampling(max_words_gen, max_sample_words=max_words_sample)
    perplexity = '{0:.2f}'.format(nGramModel.perplexity_one(generated))
    print(f"{i+1} (pp={perplexity}): {generated}")
