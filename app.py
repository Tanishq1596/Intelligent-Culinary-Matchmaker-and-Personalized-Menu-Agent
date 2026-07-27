"""Streamlit interface for the intelligent culinary matchmaker."""

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HISTORY_PATH = PROJECT_ROOT / "data" / "user_order_history.csv"
RESTAURANT_PATH = PROJECT_ROOT / "data" / "swiggy_cleaned_sample_expanded.csv"

RESTRICTION_OPTIONS = {
    "Lactose intolerant": "Lactose intolerance",
    "Gluten sensitive": "Gluten sensitivity",
    "Vegan": "Vegan",
}

ALLERGY_OPTIONS = {
    "Peanut allergy": "Peanut allergy",
    "Tree-nut allergy": "Tree-nut allergy",
    "Egg allergy": "Egg allergy",
    "Soy allergy": "Soy allergy",
}


st.set_page_config(
    page_title="Intelligent Culinary Matchmaker",
    page_icon=":material/restaurant:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f7f4; color: #17211b; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3, h4 { letter-spacing: 0 !important; }
    h1 { color: #173f31; font-size: 2.25rem !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #d8ddd9;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(23, 33, 27, 0.05);
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8ddd9;
        border-radius: 8px;
        padding: 1rem;
    }
    .safety-disclaimer {
        border-left: 4px solid #b86b24;
        background: #fff8ee;
        padding: 0.85rem 1rem;
        color: #5b3a1e;
    }
    .section-rule { border-top: 1px solid #d8ddd9; margin: 2rem 0 1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_order_history():
    return pd.read_csv(HISTORY_PATH, low_memory=False)


@st.cache_data
def load_locations():
    """Load only the two columns needed to draw the location controls."""
    return pd.read_csv(
        RESTAURANT_PATH,
        usecols=["city", "locality"],
        low_memory=False,
    )


def most_common(values):
    """Return the most frequent non-empty value."""
    cleaned = [str(value) for value in values if pd.notna(value) and str(value).strip()]
    return Counter(cleaned).most_common(1)[0][0]


def build_user_profile(
    history, user_id, city, locality, meal_time, food_preference,
    spice_preference, restrictions, allergies, user_budget,
):
    """Convert one returning user's history and current choices into model inputs."""
    user_orders = history[history["user_id"] == user_id]
    favourite_cuisine = most_common(user_orders["ordered_cuisine"])
    cuisine_orders = int((user_orders["ordered_cuisine"] == favourite_cuisine).sum())
    order_count = len(user_orders)
    day_type = "Weekend" if datetime.now().weekday() >= 5 else "Weekday"

    selected_restrictions = [
        RESTRICTION_OPTIONS[item] for item in restrictions
    ] + [
        ALLERGY_OPTIONS[item] for item in allergies
    ]

    return {
        "city": city,
        "locality": locality,
        "meal_time": meal_time,
        "dietary_preference": "Veg" if food_preference == "Vegetarian" else "Non-veg",
        "spice_preference": spice_preference,
        "restrictions": selected_restrictions,
        "user_budget": user_budget,
        "classifier_features": {
            "user_id": user_id,
            "meal_time": meal_time,
            "day_type": day_type,
            "vegetarian": "Yes" if food_preference == "Vegetarian" else "No",
            "average_previous_rating": float(user_orders["rating"].mean()),
            "most_ordered_cuisine": favourite_cuisine,
            "order_frequency": order_count,
            "spice_preference": spice_preference,
            "previous_cuisine_orders": cuisine_orders,
        },
        "spending_features": {
            "user_average_order_value": float(user_orders["order_amount"].mean()),
            "meal_time": meal_time,
            "weekday_or_weekend": day_type,
            "location": city,
            "payment_method": most_common(user_orders["payment_method"]),
            "previous_order_count": order_count,
            "preferred_cuisine": favourite_cuisine,
        },
    }


def run_recommendation(profile):
    """Run the complete agent workflow and add the regressor's raw estimate for display."""
    from agentic_ai import run_workflow
    from agentic_ai.tools import get_spending_predictor

    location_text = profile["city"]
    if profile.get("locality"):
        location_text = f"{profile['locality']}, {profile['city']}"

    result = run_workflow(f"Find a suitable meal in {location_text}.", profile)

    spending_features = dict(profile["spending_features"])
    spending_features["preferred_cuisine"] = result["predicted_cuisine"]["primary"]["cuisine"]
    result["predicted_order_value"] = get_spending_predictor().predict_expected_order_value(
        spending_features
    )
    return result


def display_safety(candidate):
    status = candidate["safety_status"]
    if status == "Likely compatible":
        st.success("No known conflict was found in the retrieved common recipe.")
    else:
        st.warning(candidate.get("reason") or "A possible dietary conflict was detected.")


def display_candidate(candidate):
    with st.container(border=True):
        st.markdown(f"#### {candidate['rank']}. {candidate['dish_name']}")

        restaurant, price, locality, rating = st.columns([2, 1, 1.4, 1])
        restaurant.markdown(f"**Restaurant**  \n{candidate['restaurant_name']}")
        price.markdown(f"**Price**  \n₹{float(candidate['price']):.0f}")
        locality.markdown(f"**Locality**  \n{candidate.get('locality') or 'Not specified'}")
        rating_value = candidate.get("restaurant_rating")
        rating.markdown(
            f"**Rating**  \n{float(rating_value):.1f}"
            if rating_value is not None
            else "**Rating**  \nNot available"
        )

        st.markdown("**Common ingredients**")
        st.write(candidate.get("common_ingredients") or "Ingredient information not listed.")

        allergens = candidate.get("allergens")
        if allergens:
            st.markdown("**Commonly reported allergens**")
            st.write(allergens)

        substitution = candidate.get("possible_substitutions")
        if substitution:
            st.markdown("**Possible substitution**")
            st.write(substitution)

        display_safety(candidate)


history = load_order_history()
locations = load_locations()

st.title("Intelligent Culinary Matchmaker")
st.caption("Personalized, grounded meal recommendations from real restaurant dishes")

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.subheader("User Profile")

profile_column, city_column, locality_column = st.columns([1.2, 1, 1.4])
with profile_column:
    user_id = st.selectbox("Order-history profile", sorted(history["user_id"].unique()))

city_list = sorted(locations["city"].dropna().astype(str).unique())
with city_column:
    city = st.selectbox("City", city_list)

city_rows = locations[locations["city"] == city]
locality_list = sorted(
    locality for locality in city_rows["locality"].dropna().astype(str).unique()
    if locality.strip()
)
with locality_column:
    if locality_list:
        locality_choice = st.selectbox("Locality", ["Any locality", *locality_list])
    else:
        locality_choice = st.selectbox(
            "Locality",
            ["Locality data unavailable"],
            disabled=True,
        )
        st.caption(f"Recommendations will search across all of {city}.")
locality = None if locality_choice == "Any locality" else locality_choice
if not locality_list:
    locality = None

meal_column, diet_column, spice_column = st.columns(3)
with meal_column:
    meal_label = st.selectbox("Meal time", ["Breakfast", "Lunch", "Snacks", "Dinner"])
    meal_time = "Evening Snack" if meal_label == "Snacks" else meal_label
with diet_column:
    food_preference = st.radio(
        "Food preference", ["Vegetarian", "Non-Vegetarian"], horizontal=True
    )
with spice_column:
    spice_preference = st.select_slider(
        "Spice preference", options=["Mild", "Medium", "Hot"], value="Medium"
    )

restriction_column, allergy_column = st.columns(2)
with restriction_column:
    restrictions = st.multiselect("Dietary restrictions", list(RESTRICTION_OPTIONS))
with allergy_column:
    allergies = st.multiselect("Allergies", list(ALLERGY_OPTIONS))

budget_toggle, budget_input = st.columns([1, 2])
with budget_toggle:
    use_explicit_budget = st.toggle("Set my own maximum budget", value=True)
with budget_input:
    budget = st.number_input(
        "Maximum budget (₹)", min_value=50, max_value=2000, value=250,
        step=25, disabled=not use_explicit_budget,
    )

if st.button(
    "Find My Meal", icon=":material/search:", type="primary", use_container_width=True
):
    if "Vegan" in restrictions and food_preference != "Vegetarian":
        st.error("Select Vegetarian when using the Vegan restriction.")
    else:
        profile = build_user_profile(
            history=history,
            user_id=user_id,
            city=city,
            locality=locality,
            meal_time=meal_time,
            food_preference=food_preference,
            spice_preference=spice_preference,
            restrictions=restrictions,
            allergies=allergies,
            user_budget=float(budget) if use_explicit_budget else None,
        )
        try:
            with st.spinner("Finding compatible dishes..."):
                st.session_state["recommendation_result"] = run_recommendation(profile)
        except Exception as error:
            st.session_state.pop("recommendation_result", None)
            st.error(f"The recommendation could not be generated: {error}")


result = st.session_state.get("recommendation_result")
if result:
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader("ML Predictions")

    primary_cuisine = result["predicted_cuisine"]["primary"]
    selected_cuisine = result["predicted_cuisine"]["selected"]
    prediction_column, spending_column, budget_column = st.columns(3)
    prediction_column.metric("Top predicted cuisine", primary_cuisine["cuisine"])
    spending_column.metric(
        "Expected order value", f"₹{result['predicted_order_value']:.0f}"
    )
    budget_source = "User limit" if result["parsed_preferences"].get("user_budget") else "Model estimate"
    budget_column.metric("Applied budget", f"₹{result['final_budget']:.0f}", budget_source)

    probabilities = result["predicted_cuisine"].get("predictions", [])
    probability_rows = [row for row in probabilities if row.get("probability") is not None]
    if probability_rows:
        chart_data = pd.DataFrame({
            "Cuisine": [row["cuisine"] for row in probability_rows],
            "Probability": [row["probability"] for row in probability_rows],
        })
        st.markdown("**Top cuisine probabilities**")
        st.bar_chart(chart_data, x="Cuisine", y="Probability", color="#2d7657")

    if selected_cuisine["cuisine"] != primary_cuisine["cuisine"]:
        st.info(
            f"No matching dishes were found for the top prediction, "
            f"{primary_cuisine['cuisine']}. The workflow used the second prediction, "
            f"{selected_cuisine['cuisine']}, for restaurant recommendations."
        )

    candidates = result.get("final_validated_candidates", [])
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    if candidates:
        st.subheader("Recommended Dishes")
        for candidate in candidates[:5]:
            display_candidate(candidate)

        st.markdown(
            '<div class="safety-disclaimer">Restaurant ingredients, preparation, and '
            'cross-contamination conditions may vary. Confirm severe allergies directly '
            'with the restaurant before ordering.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.subheader("Personalized Recommendation")
        st.markdown(result["final_response"])
    else:
        st.subheader("Recommendation Result")
        st.warning(result["final_response"])

    with st.expander("How was this recommendation generated?"):
        st.markdown(
            """
            1. The classifier predicted cuisine preference from previous orders and the current meal context.
            2. The regressor estimated the expected order value; an explicit user budget remained the hard limit.
            3. Pandas selected real dishes matching city, locality, cuisine, price, and food preference.
            4. RAG retrieved common ingredients and allergens from the culinary knowledge base.
            5. Deterministic safety rules rejected known conflicts and marked uncertain recipes with warnings.
            6. Gemini explained only the validated candidates supplied by the workflow.
            """
        )
