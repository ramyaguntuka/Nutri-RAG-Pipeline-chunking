"""Streamlit interface for the NutriRAG nutrition information assistant."""

import pandas as pd
import streamlit as st

from src.llm import GroqRequestError, MissingGroqAPIKeyError
from src.meal_planner import create_meal_plan
from src.pipeline import run_pipeline
from src.retrieval import TfidfRetriever

st.set_page_config(page_title="NutriRAG | Nutrition Assistant", page_icon="🥗", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f4fbf6 0%, #f8fafc 55%, #eef7f2 100%); }
    .hero { padding: 1.6rem 1.8rem; border-radius: 20px; color: #123b2b;
        background: linear-gradient(120deg, #d9f3e5, #eaf5ff); margin-bottom: 1rem; }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.45rem; }
    .soft-card { background: white; border: 1px solid #e6eee9; border-radius: 14px;
        padding: 1rem 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🥗 NutriRAG</h1>'
    "<div>Your friendly, source-grounded nutrition information assistant.</div></div>",
    unsafe_allow_html=True,
)

if "profile" not in st.session_state:
    st.session_state.profile = {
        "age": 25,
        "nutrition_goal": "Better energy",
        "diet_type": "Vegetarian",
        "allergies": "",
        "daily_food_budget_inr": 300,
    }
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = TfidfRetriever()

with st.sidebar:
    st.header("Your nutrition profile")
    with st.form("profile_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=int(st.session_state.profile["age"]))
        nutrition_goal = st.selectbox(
            "Nutrition goal",
            ["Weight loss", "Muscle gain", "Better energy"],
            index=["Weight loss", "Muscle gain", "Better energy"].index(
                st.session_state.profile["nutrition_goal"]
            ),
        )
        diet_type = st.selectbox(
            "Diet type",
            ["Vegetarian", "Vegan", "Non-vegetarian"],
            index=["Vegetarian", "Vegan", "Non-vegetarian"].index(
                st.session_state.profile["diet_type"]
            ),
        )
        allergies = st.text_input(
            "Food allergies (comma-separated)",
            value=str(st.session_state.profile["allergies"]),
            help="List allergens, for example: peanuts, milk",
        )
        budget = st.number_input(
            "Daily food budget (INR)",
            min_value=0,
            max_value=100000,
            value=int(st.session_state.profile["daily_food_budget_inr"]),
            step=50,
        )
        if st.form_submit_button("Save profile", type="primary"):
            st.session_state.profile = {
                "age": age,
                "nutrition_goal": nutrition_goal,
                "diet_type": diet_type,
                "allergies": allergies.strip(),
                "daily_food_budget_inr": budget,
            }
            st.success("Profile saved for this session.")

st.caption(
    "General nutrition information only—not a substitute for advice from a "
    "qualified healthcare professional."
)

chat_tab, plan_tab, about_tab = st.tabs(["💬 Ask NutriRAG", "🍽️ One-day meal plan", "ℹ️ About"])

with chat_tab:
    st.subheader("Ask a nutrition question")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Sources"):
                    for source in item["sources"]:
                        st.write(
                            f"**{source.chunk.document} — {source.chunk.section}** "
                            f"(relevance {source.score:.3f})"
                        )

    question = st.chat_input("For example: What are some vegetarian protein sources?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Looking through the nutrition notes..."):
                try:
                    result = run_pipeline(
                        question,
                        st.session_state.profile,
                        retriever=st.session_state.retriever,
                    )
                except MissingGroqAPIKeyError as exc:
                    st.error(str(exc))
                    st.info("Copy .env.example to .env, add your Groq key, then restart Streamlit.")
                except GroqRequestError as exc:
                    st.error(str(exc))
                else:
                    st.markdown(result.answer)
                    if result.sources:
                        with st.expander("Sources and retrieved sections", expanded=True):
                            for source in result.sources:
                                st.write(
                                    f"**{source.chunk.document} — {source.chunk.section}** "
                                    f"(relevance {source.score:.3f})"
                                )
                                st.caption(source.chunk.text)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        }
                    )

with plan_tab:
    st.subheader("A simple one-day meal plan")
    st.caption(
        "Illustrative meal ideas only. Calories and protein below are rough estimates, "
        "not personalized targets; actual values vary by ingredient and portion."
    )
    meals = create_meal_plan(st.session_state.profile)
    if not meals:
        st.warning(
            "The sample plan could not be filtered safely for the allergy list you entered. "
            "Please choose foods using verified ingredient and cross-contact information."
        )
    else:
        meal_table = pd.DataFrame(meals).rename(
            columns={
                "meal": "Meal",
                "food": "Idea (check ingredients)",
                "calories_estimate": "Approx. calories (estimate)",
                "protein_estimate_g": "Approx. protein (g, estimate)",
            }
        )
        st.dataframe(meal_table, hide_index=True, use_container_width=True)
    st.info(
        "Allergy screening here is a basic ingredient-name check only. It cannot verify "
        "labels, hidden ingredients, or cross-contact. Always verify food safety."
    )

with about_tab:
    st.subheader("How NutriRAG works")
    st.markdown(
        "Your question is matched to local nutrition notes using TF-IDF retrieval. "
        "When configured, Groq generates a response grounded in the matched excerpts. "
        "The source document and section are shown below each answer."
    )
    st.write("The app does not diagnose conditions or recommend medicines or treatment.")
