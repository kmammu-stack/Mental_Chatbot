# ============================================================
# INARA — AI Mental Health Companion
# Gemini LLM + Mood Sensing + A* Intervention Planning + Memory
# ============================================================

import heapq
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from google import genai
import streamlit as st

# Load API key from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

DIMENSIONS = ["mind", "body", "sound", "autonomy"]

STATE_LABELS = {
    "mind": ["Distressed", "Overwhelmed", "Neutral", "Calm", "Flourishing"],
    "body": ["Tense", "Fatigued", "Neutral", "Relaxed", "Energised"],
    "sound": ["Dysregulated", "Low", "Neutral", "Attuned", "Flow"],
    "autonomy": ["Reactive", "Guided", "Aware", "Self-directing", "Autonomous"],
}

GOAL_STATE = {"mind": 3, "body": 3, "sound": 3, "autonomy": 3}
DEFAULT_STATE = {"mind": 2, "body": 2, "sound": 2, "autonomy": 1}
MEMORY_FILE = "eq_memory.json"

INTERVENTIONS = {
    "BREATH_WORK": {
        "description": "Try box breathing: inhale for 4 seconds, hold for 4, exhale for 4, and hold for 4.",
        "delta": {"mind": 1, "body": 1, "sound": 0, "autonomy": 0},
        "cost": 1,
    },
    "SOMATIC_SCAN": {
        "description": "Pause and scan your body from head to toe. Notice three sensations without judging them.",
        "delta": {"mind": 0, "body": 2, "sound": 0, "autonomy": 1},
        "cost": 1,
    },
    "SOUND_HEALING": {
        "description": "Hum a steady, comfortable tone for about 30 seconds and notice the vibration.",
        "delta": {"mind": 1, "body": 0, "sound": 2, "autonomy": 0},
        "cost": 1,
    },
    "COGNITIVE_REFRAME": {
        "description": "Name the emotion you are experiencing, then ask what you would say to a close friend feeling this way.",
        "delta": {"mind": 2, "body": 0, "sound": 0, "autonomy": 1},
        "cost": 1,
    },
    "MOVEMENT": {
        "description": "Stand up and gently move or stretch your arms and body for about 60 seconds.",
        "delta": {"mind": 0, "body": 1, "sound": 1, "autonomy": 0},
        "cost": 1,
    },
    "JOURNALING": {
        "description": "Write freely for five minutes about what you are feeling. Do not worry about editing.",
        "delta": {"mind": 1, "body": 0, "sound": 0, "autonomy": 2},
        "cost": 2,
    },
}


def heuristic(state):
    return sum(max(0, GOAL_STATE[d] - state[d]) for d in DIMENSIONS)


def goal_reached(state):
    return all(state[d] >= GOAL_STATE[d] for d in DIMENSIONS)


def astar_plan(initial_state):
    start = tuple(initial_state[d] for d in DIMENSIONS)
    open_list = [(heuristic(initial_state), 0, start, [])]
    visited = set()

    while open_list:
        f, g, state_tuple, path = heapq.heappop(open_list)

        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        state = dict(zip(DIMENSIONS, state_tuple))

        if goal_reached(state):
            return path

        for name, action in INTERVENTIONS.items():
            new_state = {
                d: min(4, state[d] + action["delta"][d])
                for d in DIMENSIONS
            }
            new_tuple = tuple(new_state[d] for d in DIMENSIONS)

            if new_tuple not in visited:
                new_g = g + action["cost"]
                new_f = new_g + heuristic(new_state)
                heapq.heappush(
                    open_list,
                    (new_f, new_g, new_tuple, path + [name]),
                )

    return []


class MemoryManager:
    def __init__(self):
        self.long_term = self._load_long_term()
        self.short_term = []

    def _load_long_term(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("sessions", [])
                data.setdefault("intervention_scores", {})
                return data
            except (json.JSONDecodeError, OSError, ValueError):
                print("  [MEMORY] Invalid memory file — starting fresh.")

        return {"sessions": [], "intervention_scores": {}}

    def save_long_term(self, session_record):
        self.long_term["sessions"].append(session_record)

        for entry in session_record["history"]:
            name = entry["intervention"]
            score = entry["h_before"] - entry["h_after"]
            self.long_term["intervention_scores"].setdefault(name, []).append(score)

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.long_term, f, indent=2)

    def get_prior_state(self):
        if self.long_term["sessions"]:
            return self.long_term["sessions"][-1].get("final_state")
        return None

    def get_weak_interventions(self):
        weak = []
        for name, scores in self.long_term["intervention_scores"].items():
            if len(scores) >= 2 and sum(scores) / len(scores) <= 0:
                weak.append(name)
        return weak

    def log_short_term(self, name):
        self.short_term.append(name)

    def last_intervention(self):
        return self.short_term[-1] if self.short_term else None

    def intervention_count(self, name):
        return self.short_term.count(name)

    def print_memory_report(self):
        sessions = self.long_term.get("sessions", [])

        print("\n  [ MEMORY REPORT ]")
        print(f"  Total sessions : {len(sessions)}")

        if sessions:
            last = sessions[-1]
            print(f"  Last session   : {last.get('date', 'unknown')}")
            print(f"  Last state     : {last.get('final_state', {})}")

        scores = self.long_term.get("intervention_scores", {})
        if scores:
            print("  Effectiveness:")
            for name, values in scores.items():
                avg = sum(values) / len(values)
                print(f"    {name:<22} avg h drop = {avg:.2f}")


class EQCoachAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.state = dict(DEFAULT_STATE)
        self.turn = 0
        self.history = []

        # Streamlit handles all user-facing UI.

    def _label(self, dimension):
        return STATE_LABELS[dimension][self.state[dimension]]

    def _sense(self, text):
        t = text.lower()
        adjustment = {d: 0 for d in DIMENSIONS}

        if any(w in t for w in [
            "anxious", "panic", "scared", "overwhelmed",
            "stressed", "worry", "fear", "sad", "lonely"
        ]):
            adjustment["mind"] -= 1

        if any(w in t for w in [
            "better", "happy", "clear", "focused",
            "good", "great", "positive", "calm"
        ]):
            adjustment["mind"] += 1

        if any(w in t for w in [
            "tense", "tight", "pain", "tired", "exhausted",
            "heavy", "sore", "stiff", "fatigued"
        ]):
            adjustment["body"] -= 1

        if any(w in t for w in [
            "relaxed", "light", "comfortable",
            "loose", "energised", "strong"
        ]):
            adjustment["body"] += 1

        if any(w in t for w in [
            "loud", "noise", "chaotic", "ringing", "scattered"
        ]):
            adjustment["sound"] -= 1

        if any(w in t for w in [
            "quiet", "peaceful", "resonant", "humming"
        ]):
            adjustment["sound"] += 1

        if any(w in t for w in [
            "lost", "confused", "helpless",
            "stuck", "dont know", "don't know"
        ]):
            adjustment["autonomy"] -= 1

        if any(w in t for w in [
            "decided", "confident", "in control", "aware"
        ]):
            adjustment["autonomy"] += 1

        for d in DIMENSIONS:
            self.state[d] = max(
                0, min(4, self.state[d] + adjustment[d])
            )

    def _choose_intervention(self):
        plan = astar_plan(self.state)
        weak = self.memory.get_weak_interventions()
        last = self.memory.last_intervention()

        if not plan:
            return "COGNITIVE_REFRAME"

        for candidate in plan:
            if (
                candidate != last
                and self.memory.intervention_count(candidate) < 2
                and candidate not in weak
            ):
                return candidate

        return plan[0]

    def _generate_with_gemini(self, user_input, intervention_name):
        intervention = INTERVENTIONS[intervention_name]["description"]

        if not client:
            return (
                "I hear you. Let's take this one step at a time. "
                f"Try this: {intervention}"
            )

        state_summary = ", ".join(
            f"{d}={self.state[d]} ({self._label(d)})"
            for d in DIMENSIONS
        )

        prompt = f"""
You are INARA, a supportive AI mental-health companion for students.

Student message:
{user_input}

Current internal state:
{state_summary}

Selected supportive intervention:
{intervention}

Respond warmly and naturally in 3-5 sentences.
Acknowledge the student's feelings without diagnosing them.
Give the selected intervention as a simple practical suggestion.
Do not claim to be a doctor, therapist, or human.
Do not provide medication or medical diagnosis.
Do not overwhelm the user with information.

If the student indicates immediate danger, self-harm, suicide,
or danger to another person, prioritize immediate safety and
encourage them to contact local emergency services/crisis support
and a trusted person nearby.
"""

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            text = getattr(response, "text", None)
            if text:
                return text.strip()

        except Exception as exc:
            print(f"\n[Gemini Error] {exc}")

        return (
            "I hear you. You don't have to handle everything at once. "
            f"Let's try this together: {intervention}"
        )

    def respond(self, user_input):
        self.turn += 1
        h_before = heuristic(self.state)

        # 1. Detect emotional cues
        self._sense(user_input)

        # 2. Select intervention using A*
        intervention_name = self._choose_intervention()

        # 3. Generate natural response using Gemini
        response = self._generate_with_gemini(
            user_input,
            intervention_name
        )

        # 4. Apply intervention effect to internal state
        for d in DIMENSIONS:
            self.state[d] = max(
                0,
                min(
                    4,
                    self.state[d]
                    + INTERVENTIONS[intervention_name]["delta"][d]
                ),
            )

        h_after = heuristic(self.state)


        self.memory.log_short_term(intervention_name)

        self.history.append({
            "turn": self.turn,
            "user_input": user_input,
            "intervention": intervention_name,
            "h_before": h_before,
            "h_after": h_after,
            "state_after": dict(self.state),
        })

    def show_plan(self):
        plan = astar_plan(self.state)

        print("\n[ A* INTERVENTION PLAN ]")

        if not plan:
            print("Already at goal state.")
            return

        state = dict(self.state)

        for i, step in enumerate(plan):
            delta = INTERVENTIONS[step]["delta"]
            state = {
                d: min(4, state[d] + delta[d])
                for d in DIMENSIONS
            }

            print(
                f"Step {i + 1}: {step:<22} "
                f"→ h(n)={heuristic(state)}"
            )

    def show_chart(self):
        if not self.history:
            print("No session data yet.")
            return

        print("\n[ STATE HISTORY ]")

        for entry in self.history:
            print(
                f"Turn {entry['turn']} | "
                f"{entry['state_after']} | "
                f"{entry['intervention']}"
            )

    def end_session(self):
        session_record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "turns": self.turn,
            "final_state": dict(self.state),
            "history": self.history,
            "goal_reached": goal_reached(self.state),
        }

        self.memory.save_long_term(session_record)
        print("\nSession ended. See you next time.")



# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="INARA — AI Mental Health Companion",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .inara-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .inara-subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        font-size: 0.9rem;
        color: #7c2d12;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem;
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def get_agent():
    if "agent" not in st.session_state:
        st.session_state.agent = EQCoachAgent()
    return st.session_state.agent


def reset_session():
    if "agent" in st.session_state:
        st.session_state.agent.end_session()
    st.session_state.agent = EQCoachAgent()
    st.session_state.messages = []


agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧠 INARA")
    st.caption("AI Mental Health Companion")

    st.divider()

    st.markdown("### Current state")
    for dimension in DIMENSIONS:
        value = agent.state[dimension]
        label = agent._label(dimension)
        st.progress((value + 1) / 5)
        st.caption(f"**{dimension.title()}** — {label} ({value}/4)")

    st.divider()

    if st.button("🗺️ Show intervention plan", use_container_width=True):
        st.session_state.show_plan = True

    if st.button("📈 Session progress", use_container_width=True):
        st.session_state.show_progress = True

    if st.button("🧠 Memory", use_container_width=True):
        st.session_state.show_memory = True

    if st.button("🔄 New session", use_container_width=True):
        reset_session()
        st.session_state.show_plan = False
        st.session_state.show_progress = False
        st.session_state.show_memory = False
        st.rerun()

    st.divider()

    st.markdown("### Important")
    st.caption(
        "INARA is a supportive AI companion, not a doctor or therapist. "
        "It does not diagnose conditions or prescribe medication."
    )
    st.caption(
        "If you are in immediate danger or may hurt yourself or someone else, "
        "contact local emergency services or a trusted person nearby."
    )


# ---------- Header ----------
st.markdown('<div class="inara-title">🧠 INARA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="inara-subtitle">A supportive space to pause, reflect, and take the next small step.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer">INARA provides supportive conversation and practical '
    'well-being suggestions. It is not a replacement for professional mental-health care.</div>',
    unsafe_allow_html=True,
)

st.write("")


# ---------- Optional panels ----------
if st.session_state.get("show_plan", False):
    with st.expander("🗺️ A* Intervention Plan", expanded=True):
        plan = astar_plan(agent.state)
        if not plan:
            st.success("Your current state has reached the target state.")
        else:
            working_state = dict(agent.state)
            for i, step in enumerate(plan):
                delta = INTERVENTIONS[step]["delta"]
                working_state = {
                    d: min(4, working_state[d] + delta[d])
                    for d in DIMENSIONS
                }
                st.write(
                    f"**Step {i + 1}: {step.replace('_', ' ').title()}**  "
                    f"→ remaining heuristic: `{heuristic(working_state)}`"
                )
                st.caption(INTERVENTIONS[step]["description"])

if st.session_state.get("show_progress", False):
    with st.expander("📈 Session progress", expanded=True):
        if not agent.history:
            st.info("No session data yet.")
        else:
            for entry in agent.history:
                st.write(
                    f"**Turn {entry['turn']}** — "
                    f"{entry['intervention'].replace('_', ' ').title()}"
                )
                st.caption(
                    f"Heuristic: {entry['h_before']} → {entry['h_after']} | "
                    f"State: {entry['state_after']}"
                )

if st.session_state.get("show_memory", False):
    with st.expander("🧠 Memory", expanded=True):
        sessions = agent.memory.long_term.get("sessions", [])
        st.write(f"Previous sessions: **{len(sessions)}**")
        prior = agent.memory.get_prior_state()
        if prior:
            st.write(f"Last recorded state: `{prior}`")
        else:
            st.caption("No previous session has been saved yet.")


# ---------- Chat ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Tell INARA how you're feeling...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("INARA is thinking..."):
            response = agent.respond(user_input)

        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })

    st.rerun()
