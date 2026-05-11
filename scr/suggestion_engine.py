import sys
from ngram import ngram_model

sys.modules["ngram_model"] = ngram_model

from ngram.ngram_model import NGramModel


class SuggestionEngine:

    def __init__(self, ngram_model_path):
        self.ngram_model = NGramModel.load(ngram_model_path)

    def get_ngram_suggestions(self, text, top_k=5):
        context, prefix = self.ngram_model.parse_text_input(text)

        suggestions = self.ngram_model.predict_interpolated(
            context=context,
            prefix=prefix,
            top_k=top_k,
            include_scores=True,
        )

        return suggestions


if __name__ == "__main__":
    engine = SuggestionEngine("models/ngram/ngram_model.pkl")

    while True:
        text = input("\nType something: ")

        if text.lower() in ["quit", "exit", "q"]:
            break

        suggestions = engine.get_ngram_suggestions(text, top_k=3)

        print("Suggestions:")
        for word, score in suggestions:
            print(word, score)