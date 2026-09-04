"""ATS-Virus External Communication Translator Agent.

Bridges English natural-language communication between external observers,
the Virus Parent Model, and the ATS embodied agent.

Key Responsibilities:
1. Parse English feedback ("Yes", "No", positive/negative polarity) into reward signals.
2. Symbolic Token Translation (clean bidirectional mapping between concepts and tokens).
3. Bidirectional message packet parsing for parent model communication.
"""

import json
import re
from pathlib import Path
import config_rl
from ats_virus_adapter import VirusAdapter

DEFAULT_SYMBOL_DICT_PATH = config_rl.DATA_DIR / "symbolic_dictionary.json"


def _is_numeric_token(text: str) -> bool:
    """Check if token is pure digits, float, sign, or decimal fragment."""
    clean = text.strip().lstrip('+-')
    if not clean:
        return True
    # Digits or decimal numbers
    if clean.isdigit():
        return True
    try:
        float(clean)
        return True
    except ValueError:
        pass
    # Numeric fragments like 002, 03, 95
    if re.match(r"^\d+([._]\d+)?$", clean):
        return True
    return False


class SymbolicTokenMapper:
    """Handles mapping between natural language words and symbolic token codes (e.g. '0001 = Apple')."""

    def __init__(self, dict_path=None):
        self.dict_path = Path(dict_path or DEFAULT_SYMBOL_DICT_PATH)
        self.symbol_to_word = {}
        self.word_to_symbol = {}
        self._load()

    def _load(self):
        if self.dict_path.exists():
            try:
                with open(self.dict_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.symbol_to_word = data.get("symbol_to_word", {})
                    self.word_to_symbol = data.get("word_to_symbol", {})
            except Exception as e:
                print(f"[Translator] Error loading symbolic dictionary: {e}")

    def save(self):
        self.dict_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "symbol_to_word": self.symbol_to_word,
            "word_to_symbol": self.word_to_symbol,
        }
        with open(self.dict_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def register_mapping(self, symbol: str, word: str):
        """Register a mapping pair. Accepts formats like ('0001', 'Apple')."""
        clean_symbol = symbol.strip()
        clean_word = word.strip().capitalize()
        if _is_numeric_token(clean_word) or len(clean_word) <= 1:
            return  # Skip numeric debris
        self.symbol_to_word[clean_symbol] = clean_word
        self.word_to_symbol[clean_word.lower()] = clean_symbol
        self.save()

    def parse_mapping_line(self, line: str) -> tuple[str, str] | None:
        """Parses lines formatted as '0001 = Apple' or '0001: Apple'."""
        match = re.match(r"^([^\s=:]+)\s*[:=]\s*(.+)$", line.strip())
        if match:
            symbol, word = match.group(1), match.group(2)
            if not _is_numeric_token(word):
                self.register_mapping(symbol, word)
                return symbol, word
        return None

    def encode_word(self, word: str) -> str:
        """Returns symbolic token code for word or generates a deterministic fallback symbol.

        Pure-numeric tokens and single characters are filtered so they don't pollute the dictionary.
        """
        w_lower = word.strip().lower()
        if _is_numeric_token(w_lower) or len(w_lower) <= 1:
            return f"NUM:{w_lower}"
        if w_lower in self.word_to_symbol:
            return self.word_to_symbol[w_lower]
        # Hash fallback symbol: e.g., 0042$9
        val = sum(ord(c) for c in w_lower)
        symbol = f"{val:04d}${len(w_lower)}"
        self.register_mapping(symbol, word)
        return symbol

    def decode_symbol(self, symbol: str) -> str:
        """Decodes symbolic code into natural word or returns symbol if unmapped."""
        return self.symbol_to_word.get(symbol.strip(), symbol)


class ExternalTranslatorAgent:
    """External Communication Translator Agent between human feedback, Virus Parent Model, and ATS."""

    # Polarity word sets for feedback analysis
    POSITIVE_WORDS = {"yes", "yeah", "yep", "good", "correct", "right", "true", "positive", "great", "excellent", "best"}
    MID_WORDS = {"mid", "moderate", "neutral", "average", "steady", "unchanged"}
    FAIR_WORDS = {"fair", "recovering", "progress", "improving", "rising"}
    NEGATIVE_WORDS = {"no", "nope", "bad", "wrong", "false", "negative", "stop", "incorrect", "ruin", "fail", "worst"}

    def __init__(self, adapter: VirusAdapter = None, dict_path=None):
        self.adapter = adapter or VirusAdapter()
        self.mapper = SymbolicTokenMapper(dict_path=dict_path)

    def parse_feedback(self, text: str) -> float:
        """Parses English feedback into reinforcement signal:
        +1.0 for Good/Positive, +0.2 for Mid, -0.2 for Fair (improving under 0), -1.0 for Bad/Negative, 0.0 neutral.
        """
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        if "good" in tokens or len(tokens & self.POSITIVE_WORDS) > 0:
            return 1.0
        elif "mid" in tokens or len(tokens & self.MID_WORDS) > 0:
            return 0.2
        elif "fair" in tokens or len(tokens & self.FAIR_WORDS) > 0:
            return -0.2
        elif "bad" in tokens or len(tokens & self.NEGATIVE_WORDS) > 0:
            return -1.0
        return 0.0

    def parse_external_sentence(self, sentence: str) -> dict:
        """Parses external sentence into structured communication packet for Virus Parent Model."""
        feedback_val = self.parse_feedback(sentence)
        words = re.findall(r"\b[a-zA-Z]+\b", sentence)
        meaningful_words = [w for w in words if not _is_numeric_token(w) and len(w) > 1]
        encoded_tokens = [self.mapper.encode_word(w) for w in meaningful_words]

        packet = {
            "raw_sentence": sentence,
            "feedback_polarity": feedback_val,
            "encoded_symbols": encoded_tokens,
            "decoded_representation": " ".join([self.mapper.decode_symbol(s) for s in encoded_tokens]),
        }

        # Export formatted line to Virus corpus
        corpus_line = f"TRANSLATOR_MSG|{sentence}|POLARITY:{feedback_val:+0.1f}|TOKENS:{' '.join(encoded_tokens)}"
        self.adapter.export_runtime_corpus([corpus_line])

        return packet

    def format_token_definition(self, symbol: str, word: str) -> str:
        """Formats string mapping definition for future dataset training."""
        return f"{symbol} = {word}"

    def process_episode_grade(self, episode: int, grade: str, rew_per_tick: float = 0.0, score: float = 0.0) -> dict:
        """Automatically process end-of-episode grade into translator corpus packets."""
        text = f"Automatic evaluation: Episode performance was {grade}."
        return self.parse_external_sentence(text)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATS-Virus Communication & Feedback Translator")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive natural language feedback loop")
    args = parser.parse_args()

    translator = ExternalTranslatorAgent()

    if args.interactive:
        print("==========================================================")
        print("  ATS-Virus Translator Interactive Feedback Junction")
        print("  Enter English feedback or mapping commands.")
        print("  Example: 'Yes, good job exploring'")
        print("  Example: '0001 = Apple' (register token mapping)")
        print("  Type 'exit' or 'quit' to exit.")
        print("==========================================================\n")

        while True:
            try:
                user_input = input("Translator CLI> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting Translator CLI.")
                    break

                # Check if it's a mapping definition
                mapped = translator.mapper.parse_mapping_line(user_input)
                if mapped:
                    sym, word = mapped
                    print(f"  [Token Registered] Symbol: '{sym}' <-> Word: '{word}'")
                    continue

                # Parse sentence packet
                packet = translator.parse_external_sentence(user_input)
                print(f"  [Parsed Polarity] : {packet['feedback_polarity']:+1.0f} ({'POSITIVE' if packet['feedback_polarity'] > 0 else ('NEGATIVE' if packet['feedback_polarity'] < 0 else 'NEUTRAL')})")
                print(f"  [Encoded Tokens]  : {' '.join(packet['encoded_symbols'])}")
                print(f"  [Exported Corpus] : Formatted packet saved to Virus Corpus.\n")

            except KeyboardInterrupt:
                print("\nExiting Translator CLI.")
                break
    else:
        # Test feedback parsing
        print("Testing feedback parsing:")
        print("  'Yes, that is correct!' ->", translator.parse_feedback("Yes, that is correct!"))
        print("  'No, don't do that!' ->", translator.parse_feedback("No, don't do that!"))
        
        # Test token mapping
        mapper = translator.mapper
        print("\nTesting token mapper:")
        print("  Encoded 'Apple':", mapper.encode_word("Apple"))
        print("  Decoded '0001':", mapper.decode_symbol("0001"))
        print("  Encoded 'Weak Slime':", mapper.encode_word("Weak Slime"))
        print("  Decoded 'E001':", mapper.decode_symbol("E001"))
        
        # Test full sentence packet
        packet = translator.parse_external_sentence("Yes Apple is good")
        print("\nParsed sentence packet:")
        print(json.dumps(packet, indent=2))
        print("\nRun 'py -3.12 ats_virus_translator.py --interactive' to launch live feedback prompt.")
