import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path


class NGramModel:

    def __init__(self, max_n_gram=4, min_count=1, unk_token="<UNK>", start_token="<s>", end_token="</s>"):

        """ the constructor takes in 2 parameters;

            max_n_gram : 
                the maximum n-gram model we use for example max_n_gram=4 we use all n-grams up to 4-gram
            min_count: 
                    the minimum number of occurences a word needs to have to be included in the vocab we set it to 1
            """
        self.max_n_gram = max_n_gram
        self.min_count = min_count
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token

        self.vocab = set() #vocabulary
        self.word_counts = Counter() #word count unique ones
        self.total_words = 0 # number of total words

        self.counts = {n: defaultdict(Counter) for n in range(1, self.max_n_gram + 1)} # stores [n-gram model][

        self.is_trained = False



    def train(self, tokenized_sentences):
        
        sentences = [list(sentence) for sentence in tokenized_sentences]

        if not sentences:
            raise ValueError("Cannot train on an empty dataset.")

        raw_counts = Counter()

        for sentence in sentences:
            cleaned_sentence = [word.lower().strip() for word in sentence if word.strip()]
            raw_counts.update(cleaned_sentence)

        self.vocab = {word for word, count in raw_counts.items() if count >= self.min_count}

        self.vocab.add(self.unk_token)
        self.vocab.add(self.end_token)

        self.word_counts = Counter()
        self.counts = {n: defaultdict(Counter) for n in range(1, self.max_n_gram + 1)}

        for sentence in sentences:
            normalized_sentence = [self._normalize_word(word) for word in sentence if word.strip()]
            self._update_counts_from_sentence(normalized_sentence)

        self.total_words = sum(self.word_counts.values())
        self.is_trained = True



    def _update_counts_from_sentence(self, sentence):
        padded = [self.start_token] * (self.max_n_gram - 1) + list(sentence) + [self.end_token]

        for word in list(sentence) + [self.end_token]:
            self.word_counts[word] += 1

        for i in range(self.max_n_gram - 1, len(padded)):
            focus_word = padded[i]

            for n in range(1, self.max_n_gram + 1):
                context_size = n - 1

                if context_size == 0:
                    context = tuple()
                else:
                    context = tuple(padded[i - context_size:i])

                self.counts[n][context][focus_word] += 1



    def probability(self, context, word, n=4):
        self._check_trained()
        self._check_n(n)

        word = self._normalize_candidate_word(word)
        context_tuple = self._get_context_tuple(context, n)

        vocab_size = len(self.vocab)

        numerator = self.counts[n][context_tuple][word] + 1
        denominator = sum(self.counts[n][context_tuple].values()) + vocab_size

        return numerator / denominator


    def log_probability(self, context, word, n=4):
        return math.log(self.probability(context, word, n=n))


    def interpolated_probability(self, context, word, lambdas=None):
        self._check_trained()

        if lambdas is None:
            lambdas = {n: 1.0 / self.max_n_gram for n in range(1, self.max_n_gram + 1)}

        self._validate_lambdas(lambdas)

        probability_sum = 0.0

        for n, weight in lambdas.items():
            probability_sum += weight * self.probability(context, word, n=n)

        return probability_sum

    def interpolated_log_probability(self, context, word, lambdas=None):
        return math.log(self.interpolated_probability(context, word, lambdas=lambdas))

    def predict(self, context, prefix="", n=4, top_k=5, include_scores=True):
        self._check_trained()
        self._check_n(n)

        candidates = self.get_candidates(prefix)

        scored_candidates = [(word, self.probability(context, word, n=n)) for word in candidates]
        ranked = self._rank_candidates(scored_candidates, top_k=top_k)

        if include_scores:
            return ranked

        return [word for word, _ in ranked]

    def predict_interpolated(self, context, prefix="", top_k=5, lambdas=None, include_scores=True):
        self._check_trained()

        candidates = self.get_candidates(prefix)

        scored_candidates = [
            (word, self.interpolated_probability(context, word, lambdas=lambdas))
            for word in candidates
        ]

        ranked = self._rank_candidates(scored_candidates, top_k=top_k)

        if include_scores:
            return ranked

        return [word for word, _ in ranked]

    def get_candidates(self, prefix=""):
        self._check_trained()

        prefix = prefix.lower().strip()
        special_tokens = {self.start_token, self.end_token, self.unk_token}

        if prefix == "":
            return [word for word in self.vocab if word not in special_tokens]

        return [word for word in self.vocab if word not in special_tokens and word.startswith(prefix)]

    def next_word_distribution(self, context, n=4):
        self._check_trained()
        self._check_n(n)

        distribution = {}

        for word in self.vocab:
            if word in {self.start_token, self.unk_token}:
                continue

            distribution[word] = self.probability(context, word, n=n)

        return distribution

    def parse_text_input(self, text):
        if text is None:
            return [], ""

        lower_text = text.lower()

        if lower_text.endswith(" "):
            words = lower_text.strip().split()
            return words, ""

        words = lower_text.strip().split()

        if not words:
            return [], ""

        context = words[:-1]
        prefix = words[-1]

        return context, prefix

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

    
    def load(path):
        path = Path(path)

        with path.open("rb") as f:
            model = pickle.load(f)

        if not isinstance(model, NGramModel):
            raise TypeError("Loaded object is not an NGramModel.")

        return model

    def _normalize_word(self, word):
        word = word.lower().strip()

        if word in self.vocab:
            return word

        return self.unk_token

    def _normalize_candidate_word(self, word):
        word = word.lower().strip()

        if word in self.vocab:
            return word

        return self.unk_token

    def _get_context_tuple(self, context, n):
        context_size = n - 1

        if context_size == 0:
            return tuple()

        normalized_context = []

        for word in context:
            word = word.lower().strip()

            if word in self.vocab:
                normalized_context.append(word)
            else:
                normalized_context.append(self.unk_token)

        if len(normalized_context) < context_size:
            missing = context_size - len(normalized_context)
            normalized_context = [self.start_token] * missing + normalized_context

        return tuple(normalized_context[-context_size:])

    def _rank_candidates(self, scored_candidates, top_k):
        if top_k <= 0:
            return []

        ranked = sorted(
            scored_candidates,
            key=lambda item: (-item[1], -self.word_counts.get(item[0], 0), len(item[0]), item[0]),
        )

        return ranked[:top_k]

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("NGramModel must be trained before it can be used.")

    def _check_n(self, n):
        if n < 1 or n > self.max_n_gram:
            raise ValueError(f"n must be between 1 and {self.max_n_gram}, got n={n}.")

    def _validate_lambdas(self, lambdas):
        if not lambdas:
            raise ValueError("lambdas cannot be empty.")

        for n, weight in lambdas.items():
            self._check_n(n)

            if weight < 0:
                raise ValueError("Interpolation weights must be non negative.")

        total = sum(lambdas.values())

        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"interpolation weights must sum to 1. Got {total}.")

    def __repr__(self):
        return (
            f"NGramModel("
            f"max_n_gram={self.max_n_gram}, "
            f"min_count={self.min_count}, "
            f"vocab_size={len(self.vocab)}, "
            f"is_trained={self.is_trained}"
            f")"
        )