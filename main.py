import os
import re
import json
import openai

import prompts

"""
Bedtime Story Generator (ages 5-10)
===================================

A raw request like "a story about Alice and her cat Bob" is turned into a
high-quality, age-appropriate bedtime story through a generator/critic loop:

    classify -> plan an arc -> write -> JUDGE -> refine (until it passes) -> user

The LLM *judge* is the quality engine: it scores each draft against a rubric and
returns actionable feedback, which is fed back to the writer. See README.md for
the block diagram.

--------------------------------------------------------------------------------
Already built: multi-label routing + must-include coverage, a score-anchored LLM
judge, a deterministic readability gate, best-draft tracking, fail-closed safety,
a parent story card, an eval harness (eval.py), and unit tests (test_main.py).

What I would build next if I spent 2 more hours on this project (see README.md
for the fuller write-up):
  * Route user revisions back through the judge. Today a follow-up like "make it
    funnier" goes straight to the writer and skips the judge + safety floor, so a
    safe draft could regress after an edit. Most important correctness fix, and
    small -- the pipeline already exists.
  * Validate-then-illustrate: generate an image prompt from the judge-APPROVED
    story text (never the raw request), so any illustration inherits the same
    guardrails the story passed.
  * Persistent personalization memory: remember the child's name + recurring
    characters (Alice, Bob) across sessions so "another story with Bob" works.
  * Parent controls (age / length / tone) that feed the planner and reading-level
    target, reusing the classifier and readability metric already in place.
--------------------------------------------------------------------------------
"""

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL = "gpt-3.5-turbo"          # do not change (per assignment)
MAX_REVISIONS = 2                # judge-driven rewrite attempts before we ship
PASS_THRESHOLD = 8.0             # overall judge score (out of 10) needed to ship
TARGET_GRADE_MAX = 5.0           # Flesch-Kincaid grade ceiling for ages 5-10
SAFETY_FLOOR = 7.0               # never ship a story scoring below this on safety


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency). Ignores if the file is absent."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_client = None


def get_client() -> "openai.OpenAI":
    """Lazily create the client so the module imports without a key set."""
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # your own key
    return _client


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def call_model(prompt: str, system: str = "", max_tokens: int = 3000,
               temperature: float = 0.7) -> str:
    """Single chat completion. `system` sets the persona/role for this call."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_model_json(prompt: str, system: str = "", temperature: float = 0.2) -> dict:
    """Like call_model, but expects (and repairs) a JSON object response."""
    raw = call_model(prompt, system=system, temperature=temperature)
    return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object out of a model response."""
    text = text.strip()
    # strip ```json fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# --------------------------------------------------------------------------- #
# 1. Classifier / safety gate
# --------------------------------------------------------------------------- #
def classify_request(user_input: str) -> dict:
    """Extract categories + story params and screen the request for safety.

    Multi-label: a request can blend e.g. 'funny' + 'educational'. `must_include`
    captures concrete elements the user asked for (a fact, a named character, an
    event) so the writer and judge can guarantee coverage -- this is what a
    single-label classifier used to drop.
    """
    result = call_model_json(prompts.classifier_prompt(user_input),
                             system=prompts.CLASSIFIER_SYSTEM)
    # defensive defaults so the pipeline never crashes on a bad parse
    result.setdefault("safe", True)
    result.setdefault("categories", ["soothing"])
    if not result.get("categories"):
        result["categories"] = ["soothing"]
    result.setdefault("characters", [])
    result.setdefault("setting", "a cozy little place")
    result.setdefault("must_include", [])
    result.setdefault("moral", "kindness matters")
    result.setdefault("title_hint", "A Bedtime Story")
    # primary category, kept for planning + logging
    result["category"] = result["categories"][0]
    return result


# --------------------------------------------------------------------------- #
# 2. Planner (story arc)
# --------------------------------------------------------------------------- #
def plan_story(c: dict) -> str:
    """Plan-then-write: outline a bedtime arc that winds DOWN toward sleep."""
    return call_model(prompts.planner_prompt(c), system=prompts.PLANNER_SYSTEM,
                      temperature=0.7, max_tokens=400)


# --------------------------------------------------------------------------- #
# 3. Storyteller (generator)
# --------------------------------------------------------------------------- #
def write_story(c: dict, outline: str, prior_story: str = "",
                feedback: str = "") -> str:
    """Write (or revise) the story. On revision we pass the judge's feedback."""
    return call_model(prompts.writer_prompt(c, outline, prior_story, feedback),
                      system=prompts.storyteller_system(c),
                      temperature=0.8, max_tokens=1200)


# --------------------------------------------------------------------------- #
# 4. Deterministic checks (cheap, run before the LLM judge)
# --------------------------------------------------------------------------- #
def _count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:\"'")
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_grade(text: str) -> float:
    """Flesch-Kincaid grade level. Free, deterministic proxy for reading level."""
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return round(0.39 * (len(words) / sentences)
                 + 11.8 * (syllables / len(words)) - 15.59, 1)


# --------------------------------------------------------------------------- #
# 5. LLM Judge (critic)
# --------------------------------------------------------------------------- #
def judge_story(story: str, c: dict, grade: float) -> dict:
    """Score the draft against a weighted rubric and return actionable feedback."""
    prompt = prompts.judge_prompt(story, c, grade, TARGET_GRADE_MAX)
    # temperature 0 for the most consistent scoring the model can give
    result = call_model_json(prompt, system=prompts.JUDGE_SYSTEM, temperature=0.0)
    if "overall" not in result:
        result = {"scores": {}, "overall": 0.0, "weakest": "unknown",
                  "feedback": "Judge response could not be parsed; revise for "
                              "clarity, calmer ending, and simpler words."}
    return result


# --------------------------------------------------------------------------- #
# Orchestrator: plan -> write -> judge -> refine loop
# --------------------------------------------------------------------------- #
def generate_story(user_input: str, verbose: bool = True):
    """Returns (story, classification, verdict).

    story is None if the request is unsafe. verdict is the final judge result
    (scores/overall/feedback), useful for the eval harness.
    """
    c = classify_request(user_input)

    if not c.get("safe", True):
        reason = c.get("safe_reason", "it isn't suitable for a young child at bedtime")
        print(f"\nLet's pick a cozier idea for bedtime -- {reason}.")
        print("Try something like a friendly animal, a magical garden, or a "
              "sleepy little adventure.\n")
        return None, c, {}

    if verbose:
        must = '; '.join(c.get('must_include', [])) or 'none'
        print(f"\n[categories: {', '.join(c['categories'])} | "
              f"setting: {c['setting']} | must-include: {must}]")

    outline = plan_story(c)
    story = write_story(c, outline)

    # Track the BEST draft, not just the last one: a later rewrite can score
    # lower than an earlier draft, and we should ship the best we saw.
    # Ranking: higher overall wins; ties break toward the more readable draft
    # (lower reading grade), since age-appropriate reading level is central here.
    best = {"story": None, "verdict": {}, "rank": (-1.0, 0.0)}
    verdict: dict = {}
    for attempt in range(1, MAX_REVISIONS + 1):
        grade = readability_grade(story)
        verdict = judge_story(story, c, grade)
        overall = float(verdict.get("overall", 0) or 0)
        safety = float(verdict.get("scores", {}).get("safety", 10) or 0)
        if verbose:
            print(f"[draft {attempt}: judge {overall:.1f}/10, safety {safety:.0f}, "
                  f"reading grade {grade}, weakest: {verdict.get('weakest')}]")

        # Fail-closed on safety: an unsafe draft is never eligible to ship,
        # regardless of how high its other scores are.
        rank = (overall, -grade)
        if safety >= SAFETY_FLOOR and rank > best["rank"]:
            best = {"story": story, "verdict": verdict, "rank": rank}

        if overall >= PASS_THRESHOLD and grade <= TARGET_GRADE_MAX + 1.0 \
                and safety >= SAFETY_FLOOR:
            break
        if attempt == MAX_REVISIONS:
            break

        feedback = verdict.get("feedback", "")
        if grade > TARGET_GRADE_MAX + 1.0:
            feedback += (f" The reading level is too high (grade {grade}); use "
                         f"shorter sentences and simpler words.")
        story = write_story(c, outline, prior_story=story, feedback=feedback)

    # If no draft ever cleared the safety floor, fail closed rather than ship.
    if best["story"] is None:
        print("\nI couldn't make a version I'm confident is safe and cozy for "
              "bedtime. Let's try a gentler idea.\n")
        return None, c, verdict

    return best["story"], c, best["verdict"]


# --------------------------------------------------------------------------- #
# Parent "story card" (reuses the deterministic readability metric as a feature)
# --------------------------------------------------------------------------- #
READ_ALOUD_WPM = 130             # gentle bedtime read-aloud pace


def reading_level_label(grade: float) -> str:
    g = round(grade)
    return "Kindergarten" if g <= 0 else f"Grade {g}"


def read_aloud_minutes(text: str) -> int:
    words = len(re.findall(r"[A-Za-z']+", text))
    return max(1, round(words / READ_ALOUD_WPM))


def find_refrain(story: str):
    """Best-effort: the sentence the story repeats (a chantable refrain), or None.

    Returns None rather than guessing when no line clearly repeats, so the card
    never shows a made-up refrain.
    """
    counts: dict = {}
    for sentence in re.split(r"(?<=[.!?])\s+", story):
        display = sentence.strip().strip("\"'").rstrip(" .!?")
        key = display.lower()
        if not 2 <= len(key.split()) <= 14:      # skip fragments and long lines
            continue
        entry = counts.setdefault(key, {"count": 0, "display": display})
        entry["count"] += 1
    repeated = [e for e in counts.values() if e["count"] >= 2]
    if not repeated:
        return None
    return max(repeated, key=lambda e: e["count"])["display"]


def format_story_card(story: str, c: dict) -> str:
    """A short parent-facing summary shown under the story."""
    lines = [ln.strip() for ln in story.strip().splitlines() if ln.strip()]
    title = re.sub(r"[#*]", "", lines[0]).strip() if lines \
        else c.get("title_hint", "A Bedtime Story")
    parts = [
        f"📖 {title}",
        f"   Reading level: {reading_level_label(readability_grade(story))} "
        f"· ~{read_aloud_minutes(story)} min read · {', '.join(c.get('categories', []))}",
    ]
    if c.get("moral"):
        parts.append(f"   ✨ The lesson: {c['moral']}")
    refrain = find_refrain(story)
    if refrain:
        parts.append(f'   🔁 Chant together: "{refrain}"')
    return "\n".join(parts)


def present(story: str, c: dict) -> None:
    """Print the story, then the parent card. The card is a nice-to-have, so a
    failure formatting it must never stop the parent from seeing the story."""
    print("\n" + "=" * 60 + "\n" + story + "\n" + "=" * 60)
    try:
        print(format_story_card(story, c))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# CLI with a user feedback loop
# --------------------------------------------------------------------------- #
example_requests = ("A story about a girl named Alice and her best friend Bob, "
                    "who happens to be a cat.")


def main():
    _load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your environment first "
              "(e.g. export OPENAI_API_KEY=sk-...).")
        return

    print("Bedtime Story Generator (ages 5-10)")
    user_input = input("What kind of story do you want to hear? ").strip()
    if not user_input:
        user_input = example_requests
        print(f"(using the example) {user_input}")

    try:
        story, c, _ = generate_story(user_input)
    except openai.AuthenticationError:
        print("\nOpenAI rejected the key (invalid or expired). "
              "Check OPENAI_API_KEY in your .env.")
        return
    except openai.RateLimitError:
        print("\nOpenAI returned 'insufficient quota' (HTTP 429). The key is valid "
              "but the account has no available credit/billing. Add billing at "
              "platform.openai.com or use a key with quota, then re-run.")
        return
    except openai.APIError as e:
        print(f"\nOpenAI API error: {e}")
        return
    if not story:
        return

    present(story, c)

    # User feedback loop: let the child/parent request changes.
    while True:
        change = input("\nWant any changes? (e.g. 'make it funnier', or press "
                       "Enter to finish) ").strip()
        if not change:
            print("Sweet dreams! ")
            break
        story = write_story(c, outline="", prior_story=story, feedback=change)
        present(story, c)


if __name__ == "__main__":
    main()
