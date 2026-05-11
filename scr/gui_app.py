from pathlib import Path
import sys
from functools import lru_cache
from time import perf_counter

from flask import Flask, jsonify, request, render_template_string


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "scr" else CURRENT_DIR
NGRAM_DIR = PROJECT_ROOT / "scr" / "ngram"
MODEL_PATH = PROJECT_ROOT / "models" / "ngram" / "ngram_model.pkl"

sys.path.insert(0, str(NGRAM_DIR))

from ngram_model import NGramModel


app = Flask(__name__)

ngram_model = NGramModel.load(MODEL_PATH)

if not hasattr(ngram_model, "prefix_index") or not ngram_model.prefix_index:
    ngram_model._build_prefix_index()


SPECIAL_TOKENS = {
    ngram_model.start_token,
    ngram_model.end_token,
    ngram_model.unk_token,
}

VOCAB = list(ngram_model.non_special_vocab)

VOCAB.sort(key=lambda word: (-ngram_model.word_counts.get(word, 0), len(word), word))


def fast_predict_interpolated(context, prefix, top_k=5):
    candidates = ngram_model.get_candidates(prefix)

    scored_candidates = []

    for word in candidates:
        score = ngram_model.interpolated_probability(context=context, word=word)
        scored_candidates.append((word, score))

    ranked = sorted(scored_candidates, key=lambda item: (-item[1], -ngram_model.word_counts.get(item[0], 0), len(item[0]), item[0]))

    return ranked[:top_k], len(candidates)


@lru_cache(maxsize=2048)
def cached_suggest(text, top_k):
    context, prefix = ngram_model.parse_text_input(text)
    suggestions, candidate_count = fast_predict_interpolated(context=context, prefix=prefix, top_k=top_k)

    return context, prefix, suggestions, candidate_count


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>N-gram Word Predictor</title>
<style>
    :root {
        --bg: #090b10;
        --panel: rgba(18, 22, 33, 0.82);
        --panel-strong: rgba(24, 30, 44, 0.95);
        --text: #eef2ff;
        --muted: #9ca8bd;
        --border: rgba(255, 255, 255, 0.12);
        --accent: #8b5cf6;
        --accent-2: #22d3ee;
        --shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
    }

    * {
        box-sizing: border-box;
    }

    html {
        min-height: 100%;
        color-scheme: dark;
    }

    body {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        max-width: 900px;
        margin: 40px auto;
        padding: 24px;
        color: var(--text);
        background:
            radial-gradient(circle at top left, rgba(139, 92, 246, 0.26), transparent 34%),
            radial-gradient(circle at top right, rgba(34, 211, 238, 0.18), transparent 30%),
            linear-gradient(135deg, #090b10 0%, #101522 55%, #080a0f 100%);
    }

    h1 {
        margin: 0 0 8px;
        font-size: clamp(34px, 6vw, 56px);
        letter-spacing: -0.05em;
        line-height: 1;
        background: linear-gradient(135deg, #ffffff, #b9c4ff 45%, #7dd3fc);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .subtitle {
        color: var(--muted);
        margin-bottom: 28px;
        font-size: 17px;
    }

    textarea {
        width: 100%;
        height: 160px;
        font-size: 22px;
        padding: 18px 20px;
        border-radius: 20px;
        border: 1px solid var(--border);
        background: rgba(8, 12, 20, 0.76);
        color: var(--text);
        box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        resize: vertical;
        line-height: 1.5;
        caret-color: var(--accent-2);
        transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }

    textarea::placeholder {
        color: #667085;
    }

    textarea:focus {
        outline: none;
        border-color: rgba(34, 211, 238, 0.7);
        background: rgba(10, 15, 26, 0.92);
        box-shadow:
            var(--shadow),
            0 0 0 4px rgba(34, 211, 238, 0.12),
            0 0 36px rgba(139, 92, 246, 0.15);
    }

    .info {
        margin-top: 18px;
        padding: 16px 18px;
        background: var(--panel);
        border-radius: 18px;
        border: 1px solid var(--border);
        font-size: 15px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
    }

    .info-row {
        margin: 7px 0;
        color: var(--muted);
    }

    .label {
        font-weight: 700;
        color: #dbe4ff;
    }

    #contextText,
    #prefixText,
    #modeText {
        color: #f8fafc;
    }

    .status {
        color: #8ee7ff;
        font-size: 14px;
        margin-top: 10px;
    }

    .suggestions {
        margin-top: 24px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        min-height: 54px;
    }

    .suggestion-button {
        font-size: 18px;
        padding: 12px 19px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 999px;
        background:
            linear-gradient(135deg, rgba(139, 92, 246, 0.98), rgba(34, 211, 238, 0.82));
        color: white;
        cursor: pointer;
        box-shadow: 0 12px 30px rgba(34, 211, 238, 0.14);
        transition:
            transform 0.12s ease,
            box-shadow 0.12s ease,
            filter 0.12s ease;
    }

    .suggestion-button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow: 0 18px 42px rgba(139, 92, 246, 0.28);
    }

    .suggestion-button:active {
        transform: translateY(0);
    }

    .score {
        font-size: 12px;
        opacity: 0.72;
        margin-left: 8px;
    }

    .empty {
        color: var(--muted);
        margin-top: 8px;
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px dashed rgba(255, 255, 255, 0.14);
    }

    .hint {
        margin-top: 26px;
        color: var(--muted);
        font-size: 14px;
        background: var(--panel);
        padding: 15px 17px;
        border-radius: 18px;
        border: 1px solid var(--border);
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(14px);
    }

    .hint strong {
        color: #f8fafc;
    }

    kbd {
        background: rgba(255, 255, 255, 0.08);
        color: #e0f2fe;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-bottom-color: rgba(255, 255, 255, 0.28);
        border-radius: 7px;
        padding: 3px 7px;
        font-size: 12px;
        box-shadow: inset 0 -1px 0 rgba(255, 255, 255, 0.08);
    }

    ::selection {
        background: rgba(34, 211, 238, 0.35);
    }

    @media (max-width: 640px) {
        body {
            margin: 20px auto;
            padding: 18px;
        }

        textarea {
            font-size: 19px;
            border-radius: 16px;
        }

        .suggestion-button {
            font-size: 16px;
        }
    }
</style>
</head>

<body>
    <h1>N-gram Word Predictor</h1>

    <div class="subtitle">
        Suggestions update while typing using the trained n-gram model.
    </div>

    <textarea id="inputBox" placeholder="Start typing here..."></textarea>

    <div class="info">
        <div class="info-row">
            <span class="label">Context:</span>
            <span id="contextText">[]</span>
        </div>

        <div class="info-row">
            <span class="label">Current prefix:</span>
            <span id="prefixText">(empty)</span>
        </div>

        <div class="info-row">
            <span class="label">Mode:</span>
            <span id="modeText">next-word prediction</span>
        </div>

        <div class="info-row status" id="statusText">
            Ready.
        </div>
    </div>

    <div id="suggestions" class="suggestions"></div>

    <div class="hint">
        <strong>Important:</strong>
        <kbd>a very basic gui</kbd>.
        <br>
        <kbd>
    </div>

    <script>
        const inputBox = document.getElementById("inputBox");
        const suggestionsDiv = document.getElementById("suggestions");
        const contextText = document.getElementById("contextText");
        const prefixText = document.getElementById("prefixText");
        const modeText = document.getElementById("modeText");
        const statusText = document.getElementById("statusText");

        let debounceTimer = null;
        let currentAbortController = null;
        let latestRequestId = 0;

        function parseTextClientSide(text) {
            const lowerText = text.toLowerCase();

            if (lowerText.endsWith(" ")) {
                const trimmed = lowerText.trim();
                const words = trimmed.length === 0 ? [] : trimmed.split(/\\s+/);

                return {
                    context: words,
                    prefix: ""
                };
            }

            const trimmed = lowerText.trim();

            if (trimmed.length === 0) {
                return {
                    context: [],
                    prefix: ""
                };
            }

            const words = trimmed.split(/\\s+/);

            return {
                context: words.slice(0, -1),
                prefix: words[words.length - 1]
            };
        }

        function updateContextPrefixInstantly() {
            const parsed = parseTextClientSide(inputBox.value);

            contextText.textContent = JSON.stringify(parsed.context);
            prefixText.textContent = parsed.prefix === "" ? "(empty)" : parsed.prefix;

            if (parsed.prefix === "") {
                modeText.textContent = "next-word prediction";
            } else {
                modeText.textContent = "prefix completion";
            }

            return parsed;
        }

        function clearSuggestions() {
            suggestionsDiv.innerHTML = "";
        }

        function renderSuggestions(data) {
            suggestionsDiv.innerHTML = "";

            if (!data.suggestions || data.suggestions.length === 0) {
                suggestionsDiv.innerHTML = "<div class='empty'>No suggestions found.</div>";
                return;
            }

            for (const item of data.suggestions) {
                const button = document.createElement("button");
                button.className = "suggestion-button";
                button.type = "button";

                const wordSpan = document.createElement("span");
                wordSpan.textContent = item.word;

                const scoreSpan = document.createElement("span");
                scoreSpan.className = "score";
                scoreSpan.textContent = Number(item.score).toExponential(2);

                button.appendChild(wordSpan);
                button.appendChild(scoreSpan);

                button.addEventListener("click", () => {
                    insertSuggestion(item.word);
                });

                suggestionsDiv.appendChild(button);
            }
        }

        async function fetchSuggestionsNow() {
            const text = inputBox.value;
            const requestId = ++latestRequestId;

            if (currentAbortController !== null) {
                currentAbortController.abort();
            }

            currentAbortController = new AbortController();

            statusText.textContent = "Updating suggestions...";

            try {
                const response = await fetch("/suggest", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        text: text,
                        top_k: 5
                    }),
                    signal: currentAbortController.signal
                });

                if (!response.ok) {
                    throw new Error("Request failed.");
                }

                const data = await response.json();

                if (requestId !== latestRequestId) {
                    return;
                }

                contextText.textContent = JSON.stringify(data.context);
                prefixText.textContent = data.prefix === "" ? "(empty)" : data.prefix;
                modeText.textContent = data.mode;

                renderSuggestions(data);

                statusText.textContent =
                    `Updated in ${data.elapsed_ms.toFixed(1)} ms. ` +
                    `Scored ${data.candidate_count} candidates.`;

            } catch (error) {
                if (error.name === "AbortError") {
                    return;
                }

                statusText.textContent = "Could not update suggestions.";
            }
        }

        function scheduleFetchSuggestions() {
            updateContextPrefixInstantly();

            if (debounceTimer !== null) {
                clearTimeout(debounceTimer);
            }

            debounceTimer = setTimeout(() => {
                fetchSuggestionsNow();
            }, 60);
        }

        function insertSuggestion(word) {
            const text = inputBox.value;
            const parsed = parseTextClientSide(text);

            let newText = text;

            if (parsed.prefix.length > 0) {
                newText = text.slice(0, text.length - parsed.prefix.length);
            }

            if (newText.length > 0 && !newText.endsWith(" ")) {
                newText += " ";
            }

            inputBox.value = newText + word + " ";
            inputBox.focus();

            updateContextPrefixInstantly();
            clearSuggestions();

            if (debounceTimer !== null) {
                clearTimeout(debounceTimer);
            }

            fetchSuggestionsNow();
        }

        inputBox.addEventListener("input", scheduleFetchSuggestions);

        updateContextPrefixInstantly();
        fetchSuggestionsNow();
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/suggest", methods=["POST"])
def suggest():
    start_time = perf_counter()

    data = request.get_json()

    if data is None:
        data = {}

    text = data.get("text", "")
    top_k = int(data.get("top_k", 5))
    top_k = max(1, min(top_k, 10))

    context, prefix, suggestions, candidate_count = cached_suggest(text, top_k)

    elapsed_ms = (perf_counter() - start_time) * 1000

    if prefix == "":
        mode = "next-word prediction"
    else:
        mode = "prefix completion"

    result = {
        "context": context,
        "prefix": prefix,
        "mode": mode,
        "candidate_count": candidate_count,
        "elapsed_ms": elapsed_ms,
        "suggestions": [{"word": word, "score": float(score)} for word, score in suggestions],
    }

    return jsonify(result)


if __name__ == "__main__":
    print("Loaded model:")
    print(ngram_model)
    print("Vocabulary size:", len(VOCAB))
    print("Prefix index size:", len(ngram_model.prefix_index))
    app.run(debug=True, use_reloader=False, threaded=True)