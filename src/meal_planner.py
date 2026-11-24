"""A small, illustrative meal plan with conservative allergy filtering."""

from collections.abc import Mapping

MEAL_OPTIONS = {
    "Vegetarian": [
        ("Breakfast", "Oats with milk and banana", 350, 14, {"milk"}),
        ("Lunch", "Rice, dal, and cooked vegetables", 520, 20, {"none"}),
        ("Snack", "Roasted chana and a seasonal fruit", 220, 10, {"none"}),
        ("Dinner", "Tofu with vegetables and rice", 500, 24, {"soy"}),
    ],
    "Vegan": [
        ("Breakfast", "Oats with soy drink and banana", 340, 13, {"soy"}),
        ("Lunch", "Rice, lentils, and cooked vegetables", 520, 19, {"none"}),
        ("Snack", "Roasted chana and a seasonal fruit", 220, 10, {"none"}),
        ("Dinner", "Tofu with vegetables and rice", 500, 24, {"soy"}),
    ],
    "Non-vegetarian": [
        ("Breakfast", "Boiled eggs with whole-grain toast", 360, 20, {"egg", "wheat"}),
        ("Lunch", "Rice, dal, and cooked vegetables", 520, 20, {"none"}),
        ("Snack", "Roasted chana and a seasonal fruit", 220, 10, {"none"}),
        ("Dinner", "Chicken with rice and vegetables", 510, 32, {"none"}),
    ],
}


def create_meal_plan(profile: Mapping[str, object]) -> list[dict[str, object]]:
    diet = str(profile.get("diet_type") or "Vegetarian")
    allergies = {
        item.strip().casefold()
        for item in str(profile.get("allergies") or "").split(",")
        if item.strip()
    }
    plan: list[dict[str, object]] = []
    for meal, name, calories, protein, allergens in MEAL_OPTIONS.get(
        diet, MEAL_OPTIONS["Vegetarian"]
    ):
        if allergies.intersection(allergens):
            continue
        plan.append(
            {
                "meal": meal,
                "food": name,
                "calories_estimate": calories,
                "protein_estimate_g": protein,
            }
        )
    return plan
