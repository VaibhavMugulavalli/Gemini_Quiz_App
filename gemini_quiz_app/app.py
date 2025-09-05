import os
import re
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, make_response
)

# Optional: load environment variables from a .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Google Gemini (generative AI) SDK
import google.generativeai as genai


# -----------------------------------------------------------------------------
# Basic config (self-contained; works locally and on EC2 without extras)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Logging to console (you’ll see errors with `python app.py` or in `nohup.out`)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Secrets / API key from env (set on EC2 before running)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    app.logger.warning("No GEMINI_API_KEY / GOOGLE_API_KEY set; quiz generation will fail.")
else:
    genai.configure(api_key=API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # change if you like


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _extract_json(text: str) -> Dict:
    """Extract JSON from model output; tolerant of ```json fences and trailing commas."""
    if not text:
        raise ValueError("Empty response from model")

    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not locate JSON in model output")
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate2)


def build_quiz_prompt(topic: str, difficulty: str, n_questions: int, subtopics: List[str]) -> str:
    subtopics_str = ", ".join([s.strip() for s in subtopics if s.strip()]) if subtopics else ""
    return f"""
You are a strict quiz generator for **Machine Learning / AI / Computer Science**.
Create **{n_questions}** **{difficulty}**-level multiple-choice questions on: "{topic}".

Rules:
- Exactly 4 options per question (A–D), only ONE correct option.
- Provide a short explanation for the correct answer.
- Tag each question with a single concise subtopic (e.g., "CNNs", "Backprop", "SVM", "Time Complexity", "OS Scheduling", "CAP theorem").
- Questions must be self-contained and accurate.
- Vary style (concepts, quick math, small code) as appropriate.
- Keep explanations crisp (1–3 sentences).
- Prefer the user-specified subtopics.

Return **ONLY JSON** with this schema:
{{
  "quiz_title": "string",
  "questions": [
    {{
      "id": 1,
      "subtopic": "string",
      "question": "string",
      "options": {{"A": "string","B": "string","C": "string","D": "string"}},
      "correct": "A",
      "explanation": "string"
    }}
  ]
}}

Bias questions toward: [{subtopics_str}]
""".strip()


def generate_quiz(topic: str, difficulty: str, n_questions: int, subtopics: List[str]) -> Dict:
    if not API_KEY:
        raise RuntimeError("Missing Gemini API key")
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = build_quiz_prompt(topic, difficulty, n_questions, subtopics)
    resp = model.generate_content(prompt)
    data = _extract_json(resp.text)

    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Model returned invalid quiz JSON")

    data["questions"] = data["questions"][:n_questions]
    for i, q in enumerate(data["questions"], start=1):
        q["id"] = i
    return data


def analyze_results(quiz: Dict, user_answers: Dict[str, str]) -> Tuple[int, int, List[Dict], Dict[str, Dict[str, int]]]:
    """
    Returns: score, total, per-question list (with explanations), and subtopic stats.
    """
    per_q = []
    subtopic_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_count = 0

    for q in quiz["questions"]:
        qid = str(q["id"])
        user_ans = user_answers.get(qid)
        is_correct = (user_ans == q["correct"])
        if is_correct:
            correct_count += 1

        sub = q.get("subtopic", "General")
        subtopic_stats[sub]["total"] += 1
        if is_correct:
            subtopic_stats[sub]["correct"] += 1

        per_q.append({
            "id": q["id"],
            "question": q["question"],
            "options": q["options"],
            "correct": q["correct"],
            "user": user_ans,
            "explanation": q.get("explanation", ""),
            "subtopic": sub,
            "is_correct": is_correct
        })

    return correct_count, len(quiz["questions"]), per_q, dict(subtopic_stats)


def build_coach_feedback(subtopic_stats: Dict[str, Dict[str, int]]) -> Dict[str, List[Dict]]:
    """
    Pure-Python feedback: strengths (>=80% acc & >=2 qs) and focus areas
    (<50% acc OR only 1 q and incorrect). Each focus area has 3 actionable tips.
    """
    strengths, focus = [], []
    for sub, st in subtopic_stats.items():
        total = max(1, st["total"])
        acc = 100.0 * st["correct"] / total

        if st["total"] >= 2 and acc >= 80:
            strengths.append({"subtopic": sub, "note": "Great grasp—continue with tougher, mixed problems."})
        elif (st["total"] == 1 and st["correct"] == 0) or acc < 50:
            focus.append({
                "subtopic": sub,
                "pointers": [
                    "Revisit definitions & key formulas; write a 5-line summary.",
                    "Solve 3–5 targeted practice problems in this area.",
                    "Explain the concept aloud or teach a friend to solidify gaps."
                ]
            })
    return {"strengths": strengths, "focus": focus}


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    topic = request.form.get("topic", "").strip()
    difficulty = request.form.get("difficulty", "Medium").strip()
    n_questions = int(request.form.get("n_questions", "8"))
    subtopics_raw = request.form.get("subtopics", "").strip()
    subtopics = [s.strip() for s in subtopics_raw.split(",")] if subtopics_raw else []

    if not topic:
        flash("Please enter a topic.")
        return redirect(url_for("index"))

    try:
        quiz = generate_quiz(topic, difficulty, n_questions, subtopics)
    except Exception as e:
        app.logger.exception("Failed to generate quiz")
        return render_template("error.html", message=f"Failed to generate quiz: {e}")

    session["topic"] = topic
    session["difficulty"] = difficulty
    session["quiz"] = quiz
    return redirect(url_for("quiz"))


@app.route("/quiz", methods=["GET"])
def quiz():
    quiz = session.get("quiz")
    if not quiz:
        flash("No quiz in session. Start again.")
        return redirect(url_for("index"))
    return render_template("quiz.html", quiz=quiz, topic=session.get("topic"), difficulty=session.get("difficulty"))


@app.route("/submit", methods=["POST"])
def submit():
    quiz = session.get("quiz")
    if not quiz:
        flash("Session expired. Please start again.")
        return redirect(url_for("index"))

    user_answers = {str(q["id"]): request.form.get(f"q{q['id']}") for q in quiz["questions"]}
    score, total, per_q, subtopic_stats = analyze_results(quiz, user_answers)
    feedback = build_coach_feedback(subtopic_stats)

    return render_template(
        "results.html",
        score=score, total=total,
        per_q=per_q,
        subtopic_stats=subtopic_stats,
        strengths=feedback.get("strengths", []),
        focus=feedback.get("focus", []),
        topic=session.get("topic"), difficulty=session.get("difficulty"),
    )


@app.route("/healthz", methods=["GET"])
def healthz():
    return make_response({"status": "ok"}, 200)


@app.errorhandler(Exception)
def handle_any_error(e):
    app.logger.exception("Unhandled exception")
    try:
        return render_template("error.html", message=str(e)), 500
    except Exception:
        return make_response({"error": "Internal Server Error", "detail": str(e)}, 500)


# -----------------------------------------------------------------------------
# Entrypoint (built-in Flask server)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind to all interfaces so you can hit it via the EC2 public IP.
    #host = 
    port = int(os.getenv("PORT", "5000"))  # open this port in the EC2 security group
    app.run( port=8000, debug=True, use_reloader=False)
