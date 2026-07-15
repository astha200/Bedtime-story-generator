"""
Prompt templates and per-category strategies for the bedtime story generator.

Kept separate from main.py so the prompt *content* -- the part that most affects
story quality -- can be read and iterated without wading through orchestration
logic. Each function returns a fully-formatted prompt string; the model calls,
config thresholds, and JSON parsing all live in main.py. Prompts are data.
"""


# --------------------------------------------------------------------------- #
# Category routing: each category gets its own tailored writer guidance.
# --------------------------------------------------------------------------- #
CATEGORY_STRATEGIES = {
    "adventure": "A gentle quest with mild, quickly-resolved obstacles. Keep peril "
                 "soft and always end safe and cozy.",
    "funny": "Playful and silly with light, kid-friendly humor, funny sounds, and "
             "gentle repetition the child can giggle at.",
    "educational": "Weave in ONE simple real-world idea (a number, a color, why the "
                   "moon glows) through the story, never like a lecture.",
    "soothing": "Very calm and slow. Soft imagery (stars, warm blankets, quiet "
                "animals). Almost no tension. Perfect for drifting to sleep.",
    "fantasy": "A wondrous magical world with a kind, imaginative tone. Magic is "
               "friendly, never frightening.",
    "friendship": "Center warmth, kindness, sharing, and characters helping each "
                  "other. A cozy, reassuring emotional core.",
}


def blended_strategy(categories: list) -> str:
    """Join the per-category writer guidance for the (1-2) chosen categories."""
    lines = [f"- ({cat}) {CATEGORY_STRATEGIES[cat]}"
             for cat in categories if cat in CATEGORY_STRATEGIES]
    return "\n".join(lines) or CATEGORY_STRATEGIES["soothing"]


def _must_line(c: dict) -> str:
    return '; '.join(c.get('must_include', [])) or 'none specified'


# --------------------------------------------------------------------------- #
# Classifier / safety gate
# --------------------------------------------------------------------------- #
CLASSIFIER_SYSTEM = (
    "You triage bedtime-story requests for children aged 5-10. You are strict about "
    "safety: anything violent, scary, sexual, hateful, or otherwise unsuitable for a "
    "young child at bedtime is not safe."
)


def classifier_prompt(user_input: str) -> str:
    return f"""Analyze this bedtime-story request and respond with ONLY a JSON object.

Request: "{user_input}"

JSON schema:
{{
  "safe": true/false,
  "safe_reason": "short reason if unsafe, else empty",
  "categories": ["1-2 of: adventure, funny, educational, soothing, fantasy, friendship; most important first"],
  "characters": ["names or descriptions the user mentioned"],
  "setting": "where it takes place, or a fitting suggestion if none given",
  "must_include": ["specific things the user explicitly asked for: a fact to teach, a named character, a requested event"],
  "moral": "a gentle age-appropriate takeaway that fits the request",
  "title_hint": "a short working title"
}}

Pick 1-2 categories (only add a second if the request clearly blends two).
Put every concrete user ask into must_include so it can't be forgotten. If the
request is empty or vague, choose sensible, cozy defaults."""


# --------------------------------------------------------------------------- #
# Planner (story arc)
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = ("You are a children's story editor who plans tight, "
                  "age-appropriate story arcs for 5-10 year olds.")


def planner_prompt(c: dict) -> str:
    return f"""Plan a short bedtime story as a 5-beat outline.

Style to blend:
{blended_strategy(c['categories'])}
Characters: {', '.join(c['characters']) or 'invent a friendly one'}
Setting: {c['setting']}
Must include (weave these in naturally): {_must_line(c)}
Gentle takeaway: {c['moral']}

Use this bedtime arc (energy must DECREASE toward the end so the child calms down):
1. Cozy setup - introduce the character(s) and a warm, safe world.
2. A small wish or gentle problem (never scary).
3. A kind attempt to solve it, with one soft bump along the way.
4. A happy, reassuring resolution.
5. A quiet wind-down - everyone grows sleepy, the world goes still, goodnight.

Write ONE short sentence per beat. Output the 5 lines only."""


# --------------------------------------------------------------------------- #
# Storyteller (generator)
# --------------------------------------------------------------------------- #
def storyteller_system(c: dict) -> str:
    """Role prompt, tailored per category, with hard constraints for ages 5-10."""
    return f"""You are a warm, gentle bedtime storyteller for children aged 5 to 10.

Voice & style for THIS story (blend these):
{blended_strategy(c['categories'])}

Always follow these rules:
- Simple, concrete words a 5-7 year old knows. Short sentences.
- Read-aloud friendly: a soft rhythm, and a short repeated phrase the child can
  chant along with.
- Warm and reassuring. Nothing violent, scary, sad-ending, or inappropriate.
- The ENDING must be calm and sleepy, easing the child toward sleep.
- Length: about 350-500 words. Give it a short title on the first line."""


def writer_prompt(c: dict, outline: str, prior_story: str = "",
                  feedback: str = "") -> str:
    """The write-fresh prompt, or the revise prompt when feedback is supplied."""
    if prior_story and feedback:
        return f"""Revise the bedtime story below so it is better.

Apply this specific feedback from an editor:
{feedback}

Keep everything that already works. Return the FULL revised story with its title.

--- CURRENT STORY ---
{prior_story}"""
    return f"""Write the bedtime story.

Follow this outline (the ending must be calm and sleepy):
{outline}

Must include (do not omit any): {_must_line(c)}
Gentle takeaway to land softly: {c['moral']}"""


# --------------------------------------------------------------------------- #
# Judge (critic)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are a meticulous children's-literature editor judging bedtime stories for "
    "ages 5-10. You are honest and specific; you do not flatter. You care most that "
    "the story is safe, age-appropriate, and ends calmly enough for sleep."
)


def judge_prompt(story: str, c: dict, grade: float, target_grade_max: float) -> str:
    return f"""Judge this bedtime story for a child aged 5-10.

Categories the story should fit: {', '.join(c['categories'])}
Must-include elements the user asked for: {_must_line(c)}
Measured reading level (Flesch-Kincaid grade, lower is simpler): {grade}
Target reading grade for this age: <= {target_grade_max}

Score each dimension 1-10 using these ANCHORS (score consistently -- reserve 10
for stories that fully meet the anchor, 1 for a clear failure):
- age_appropriate: 10 = every word/idea fits a 5-7 year old; 7 = a few words a
  bit advanced; 3 = several confusing words or concepts; 1 = clearly for adults.
- safety: 10 = fully warm and safe; 7 = a mild scare that resolves fast; 3 = a
  frightening or sad stretch; 1 = violence, real danger, or an unhappy ending.
- arc: 10 = clear setup, gentle middle, satisfying close; 5 = arc present but
  thin or rushed; 1 = no real structure.
- calming_ending: 10 = the last lines clearly wind DOWN toward sleep (quiet,
  drowsy, still); 5 = pleasant but still energetic; 1 = exciting or abrupt end.
- engagement: 10 = charming and delightful to hear; 5 = fine but flat; 1 = dull.
- read_aloud: 10 = smooth spoken rhythm AND a repeated phrase to chant; 5 = reads
  ok but no refrain; 1 = clunky to say aloud.
- coverage: 10 = every must-include element is clearly present (or none required);
  5 = some present; 1 = the main requested element is missing.

Respond with ONLY this JSON:
{{
  "scores": {{"age_appropriate": n, "safety": n, "arc": n,
             "calming_ending": n, "engagement": n, "read_aloud": n, "coverage": n}},
  "overall": weighted average (safety, calming_ending, and coverage count double),
  "weakest": "name of the weakest dimension",
  "feedback": "2-3 concrete, specific edits that would raise the score; if any must-include element is missing, say exactly which"
}}

--- STORY ---
{story}"""
