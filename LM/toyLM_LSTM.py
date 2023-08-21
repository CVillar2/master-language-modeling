###################
# INPUTS
###################

# Hyper-parameters
max_seq_length = 10
embedding_dim = 128
lstm_units = 1024
batch_size = 256
epochs = 20
max_words = None
eos_token = 'eeeooosss'
oov_token = '<unk>'
max_words_gen = 30
max_words_sample = 50
number_to_generate = 3

dataset_base = 'HerMajestySpeechesDataset'
#dataset_base = 'Hobbit'
#dataset_base = 'LOTR'
train_set_file = dataset_base + '/train.txt'
dev_set_file = dataset_base + '/dev.txt'
test_set_file = dataset_base + '/test.txt'


###################
# IMPORTS
###################

from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np
import math, random


###################
# PRE-PROCESSING
###################

# Read the train set
train_file = open(train_set_file, 'r', encoding="utf8")
texts = train_file.readlines()
for i in range(len(texts)):
    texts[i] = texts[i] + " " + eos_token

# define the tokenizer
tokenizer = Tokenizer(oov_token=oov_token, num_words=max_words)
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1 if max_words is None else min(tokenizer.num_words, len(tokenizer.word_index) + 1)
eos_index = tokenizer.word_index[eos_token]
print(f"index('<EOS>') = {eos_index}")

# convert the text to a sequence of tokens
sequences = tokenizer.texts_to_sequences(texts)
sequences = [s for s in sequences if len(s) > 0]

# generate n-grams from the sequences
ngrams = []
for sequence in sequences:
    for i in range(0, len(sequence)):
        ngrams.append(sequence[max(i-max_seq_length,0):i+1])

# split the n-grams into input and output
#x = ngrams[:,:-1]
x = [sublist[:-1] if len(sublist) > 1 else [] for sublist in ngrams]
#y = ngrams[:,-1]
y = [sublist[-1] if len(sublist) > 0 else [] for sublist in ngrams]

# convert the output to a one-hot encoding
y = to_categorical(y, num_classes=vocab_size)

# pad the input sequences to a fixed length
x = pad_sequences(x, maxlen=max_seq_length)


###################
# MODEL
###################

# define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length, mask_zero=True))
model.add(LSTM(lstm_units))
#model.add(Dropout(0.5))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


###################
# TRAIN
###################

model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.0)


###################
# EVALUATE
###################

def perplexity(model, test_set, n, tokenizer):
    perp_list = []
    for sentence in test_set:
        # Pad the sentence with start and end tokens
        sentence = sentence + " " + eos_token
        sentence = tokenizer.texts_to_sequences([sentence])[0]
        N = len(sentence)
        if N == 0:
            continue
        prob_product = 1
        # Iterate over each n-gram in the sentence
        contexts = []
        targets = []
        for i in range(0, len(sentence)):
            contexts.append(sentence[max(i-n,0):i])
            targets.append(sentence[i])
        padded_ngrams = pad_sequences(contexts, maxlen=n)
        # Calculate the probability of the n-gram using the model
        probs = model.predict(np.array(padded_ngrams), verbose=0)
        for i in range(len(probs)):
            pi = probs[i]
            prob = pi[targets[i]]
            prob_product *= 1/prob
        # Calculate the perplexity for this sentence
        if N > 0 and prob_product > 0:
            perplexity = prob_product**(1/N)
            perp_list.append(perplexity)
    # Calculate the average perplexity over all sentences
    if len(perp_list) > 0:
        avg_perplexity = sum(perp_list) / len(perp_list)
    else:
        avg_perplexity = float('inf')
    return avg_perplexity


print("")
# Perplexity of the dev set
dev_file = open(dev_set_file, 'r', encoding="utf8")
dev_dataset = dev_file.readlines()
perplexity_dev = perplexity(model, dev_dataset, max_seq_length, tokenizer)
print("Perplexity DEV:", '{0:.2f}'.format(perplexity_dev))

# Perplexity of the test set
test_file = open(test_set_file, 'r', encoding="utf8")
test_dataset = test_file.readlines()
perplexity_test = perplexity(model, test_dataset, max_seq_length, tokenizer)
print("Perplexity TEST:", '{0:.2f}'.format(perplexity_test))

# Perplexity of first sentences of DEV set
for i in range(2):
    pp = perplexity(model, [dev_dataset[i]], max_seq_length, tokenizer)
    fpp = '{0:.2f}'.format(pp)
    print(f'\nPERPLEXITY = {fpp}: "{dev_dataset[i]}"')


###################
# GENERATE
###################

# define a function to generate text from the model
def generate_text(seed_text, next_words, model, tokenizer, n, sample=False, max_sample_words=None):
    gen_text = seed_text
    for _ in range(next_words):
        # convert the seed text to a sequence of tokens
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        # pad the sequence to length n-1
        padded_sequence = pad_sequences([sequence], maxlen=n)
        # predict the next token
        pred = model.predict(padded_sequence, verbose=0)[0]
        if sample:
            pred_trunc = list(pred)
            indexes = list(range(0, len(pred)))
            if max_sample_words is not None:
                sorted_pairs = sorted(zip(pred_trunc, indexes), reverse=True)
                sorted_pairs = sorted_pairs[:max_sample_words]
                pred_trunc, indexes = zip(*sorted_pairs)
            samples = random.choices(indexes, weights=pred_trunc, k=1)
            next_token = tokenizer.index_word[samples[0]]
        else:
            # select the most likely token
            next_token = tokenizer.index_word[np.argmax(pred)]
        # add the next token to the seed text
        if len(sequence) < n:
            seed_text += " " + next_token
        else:
            text_seq = tokenizer.sequences_to_texts([sequence[1:]])[0]
            seed_text = text_seq + " " + next_token
        if next_token == eos_token:
            break
        gen_text += " " + next_token
    return gen_text.strip()


print(f"\nGENERATE MAX {max_words_gen} WORDS / MOST LIKELY")
generated_text = generate_text("", max_words_gen, model, tokenizer, max_seq_length)
pp = '{0:.2f}'.format(perplexity(model, [generated_text], max_seq_length, tokenizer))
print(f"1 (pp={pp}): {generated_text}")

print(f"\nGENERATE MAX {max_words_gen} WORDS / SAMPLING")
for i in range(number_to_generate):
    generated_text = generate_text("", max_words_gen, model, tokenizer, max_seq_length, sample=True)
    pp = '{0:.2f}'.format(perplexity(model, [generated_text], max_seq_length, tokenizer))
    print(f"{i+1} (pp={pp}): {generated_text}")

print(f"\nGENERATE MAX {max_words_gen} WORDS / SAMPLING {max_words_gen} MOST LIKELY")
for i in range(number_to_generate):
    generated_text = generate_text("", max_words_gen, model, tokenizer, max_seq_length, sample=True, max_sample_words=max_words_sample)
    pp = '{0:.2f}'.format(perplexity(model, [generated_text], max_seq_length, tokenizer))
    print(f"{i+1} (pp={pp}): {generated_text}")
