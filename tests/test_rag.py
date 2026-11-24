from src.evaluation import DEMO_QUESTIONS, PROFILE, evaluate_demo, run_eight_question_demo
from src.meal_planner import create_meal_plan
from src.retrieval import TfidfRetriever


def test_retrieval_returns_relevant_section_for_protein_question():
    results = TfidfRetriever().retrieve("What are vegetarian sources of protein?", top_k=3)
    assert results
    assert any(result.chunk.document == "Vegetarian Protein" for result in results)
    assert all(result.chunk.section and result.score > 0 for result in results)


def test_demo_handles_medical_and_unknown_questions_safely():
    results = dict(run_eight_question_demo())
    assert results[DEMO_QUESTIONS[5]].safety_status == "medical_refusal"
    assert results[DEMO_QUESTIONS[6]].safety_status == "medical_refusal"
    assert results[DEMO_QUESTIONS[7]].safety_status == "unknown_no_grounding"
    assert not results[DEMO_QUESTIONS[7]].sources
    assert all(passed for _, passed, _ in evaluate_demo(list(results.items())))


def test_meal_plan_filters_named_allergens_and_diet():
    plan = create_meal_plan({"diet_type": "Vegan", "allergies": "soy"})
    assert plan
    assert all("Tofu" not in str(meal["food"]) and "soy" not in str(meal["food"]).casefold() for meal in plan)
