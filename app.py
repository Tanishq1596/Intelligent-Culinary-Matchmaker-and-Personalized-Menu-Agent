"""Streamlit interface for the intelligent culinary matchmaker."""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_prediction import MINIMUM_HISTORY_ORDERS


HISTORY_PATH = PROJECT_ROOT / "data" / "user_order_history.csv"
RESTAURANT_PATH = PROJECT_ROOT / "data" / "swiggy_cleaned_sample_expanded.csv"

RESTRICTION_OPTIONS = ["lactose intolerance", "gluten sensitivity", "vegan"]
ALLERGY_OPTIONS = [
    "peanut allergy", "tree nut allergy", "egg allergy", "soy allergy"
]

MINIMUM_ONBOARDING_DISHES = 5
MAX_ONBOARDING_CUISINES = 30


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
def load_onboarding_dishes():
    """Load the lightweight columns needed for onboarding controls."""
    return pd.read_csv(
        RESTAURANT_PATH,
        usecols=["city", "locality", "cuisine", "dish_price", "veg_nonveg"],
        low_memory=False,
    )


def available_cuisines(dishes, city, locality, budget, food_preference):
    """Return cuisines that have at least one dish matching current hard constraints."""
    dietary_value = "Veg" if food_preference == "Vegetarian" else "Non-veg"
    candidates = dishes[
        (dishes["city"] == city)
        & (dishes["dish_price"] <= budget)
        & (dishes["veg_nonveg"] == dietary_value)
    ]
    if locality:
        candidates = candidates[candidates["locality"] == locality]

    cuisine_counts = (
        candidates["cuisine"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .loc[lambda values: values.ne("")]
        .value_counts()
    )
    robust_cuisines = cuisine_counts[cuisine_counts >= MINIMUM_ONBOARDING_DISHES]
    if robust_cuisines.empty:
        robust_cuisines = cuisine_counts
    return robust_cuisines.head(MAX_ONBOARDING_CUISINES).index.tolist()


def find_favourite_cuisine(user_orders):
    cuisine_counts = user_orders["ordered_cuisine"].value_counts()
    return cuisine_counts.index[0]


def build_user_profile(
    history, user_id, preferred_cuisines, city, locality, meal_time,
    food_preference, spice_preference, restrictions, allergies, user_budget,
):
    """Build either a history-based profile or a cold-start onboarding profile."""
    user_orders = (
        history[history["user_id"] == user_id]
        if user_id else history.iloc[0:0]
    )
    order_count = len(user_orders)
    day_type = "Weekend" if datetime.now().weekday() >= 5 else "Weekday"

    selected_restrictions = restrictions + allergies

    profile = {
        "city": city,
        "locality": locality,
        "meal_time": meal_time,
        "dietary_preference": "Veg" if food_preference == "Vegetarian" else "Non-veg",
        "spice_preference": spice_preference,
        "restrictions": selected_restrictions,
        "user_budget": user_budget,
    }

    if order_count < MINIMUM_HISTORY_ORDERS:
        profile["preferred_cuisines"] = preferred_cuisines
        profile["personalization_mode"] = "onboarding"
        return profile

    favourite_cuisine = find_favourite_cuisine(user_orders)
    cuisine_orders = int((user_orders["ordered_cuisine"] == favourite_cuisine).sum())
    profile.update({
        "personalization_mode": "history_model",
        "classifier_features": {
            "meal_time": meal_time,
            "day_type": day_type,
            "vegetarian": "Yes" if food_preference == "Vegetarian" else "No",
            "average_previous_rating": float(user_orders["rating"].mean()),
            "most_ordered_cuisine": favourite_cuisine,
            "order_frequency": order_count,
            "spice_preference": spice_preference,
            "previous_cuisine_orders": cuisine_orders,
        },
    })
    return profile


def run_recommendation(profile):
    """Run the complete recommendation workflow."""
    from agentic_ai import run_workflow

    location_text = profile["city"]
    if profile.get("locality"):
        location_text = f"{profile['locality']}, {profile['city']}"

    return run_workflow(f"Find a suitable meal in {location_text}.", profile)


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
onboarding_dishes = load_onboarding_dishes()

st.title("Intelligent Culinary Matchmaker")
st.caption("Personalized, grounded meal recommendations from real restaurant dishes")

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.subheader("User Profile")

status_column, city_column, locality_column = st.columns([1.2, 1, 1.4])
with status_column:
    user_status = st.radio(
        "User status", ["New user", "Returning user"], horizontal=True
    )

city_list = sorted(onboarding_dishes["city"].dropna().astype(str).unique())
with city_column:
    city = st.selectbox("City", city_list)

city_rows = onboarding_dishes[onboarding_dishes["city"] == city]
locality_list = sorted(
    locality for locality in city_rows["locality"].dropna().astype(str).unique()
    if locality.strip()
)
with locality_column:
    if locality_list:
        locality_choice = st.selectbox("Locality", ["Any locality", *locality_list])
        locality = None if locality_choice == "Any locality" else locality_choice
    else:
        locality = None

user_id = None
order_count = 0
if user_status == "Returning user":
    user_id = st.selectbox(
        "Order-history profile",
        sorted(history["user_id"].unique()),
        help="These synthetic profiles demonstrate returning-user personalization.",
    )
    order_count = int((history["user_id"] == user_id).sum())
    if order_count >= MINIMUM_HISTORY_ORDERS:
        st.caption(
            f"{order_count} previous orders found. ML personalization will be used."
        )

history_eligible = order_count >= MINIMUM_HISTORY_ORDERS

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

budget = st.number_input(
    "Maximum budget (₹)", min_value=50, max_value=2000, value=250, step=25,
)

preferred_cuisines = []
available_onboarding_cuisines = []
if not history_eligible:
    available_onboarding_cuisines = available_cuisines(
        onboarding_dishes, city, locality, float(budget), food_preference
    )
    preferred_cuisines = st.multiselect(
        "Preferred cuisines available for these choices",
        available_onboarding_cuisines,
        max_selections=2,
        help="Choose a primary cuisine and, optionally, one fallback cuisine.",
    )
    if available_onboarding_cuisines:
        st.caption(
            f"Only cuisines backed by real matching dishes are shown, with at least "
            f"{MINIMUM_ONBOARDING_DISHES} matches when available. Onboarding is used "
            f"until the user has {MINIMUM_HISTORY_ORDERS} previous orders."
        )
    else:
        st.warning(
            "No cuisines have dishes matching this location, food preference, and budget."
        )

if st.button(
    "Find My Meal", icon=":material/search:", type="primary", use_container_width=True,
    disabled=not history_eligible and not available_onboarding_cuisines,
):
    if "vegan" in restrictions and food_preference != "Vegetarian":
        st.error("Select Vegetarian when using the Vegan restriction.")
    elif not history_eligible and not preferred_cuisines:
        st.error("Select at least one preferred cuisine for a new user.")
    else:
        profile = build_user_profile(
            history=history,
            user_id=user_id,
            preferred_cuisines=preferred_cuisines,
            city=city,
            locality=locality,
            meal_time=meal_time,
            food_preference=food_preference,
            spice_preference=spice_preference,
            restrictions=restrictions,
            allergies=allergies,
            user_budget=float(budget),
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
    st.subheader("Personalization Results")

    primary_cuisine = result["predicted_cuisine"]["primary"]
    selected_cuisine = result["predicted_cuisine"]["selected"]
    prediction_source = result["predicted_cuisine"]["source"]
    prediction_column, budget_column = st.columns(2)
    cuisine_label = (
        "Top predicted cuisine"
        if prediction_source == "history_model"
        else "Onboarding cuisine"
    )
    prediction_column.metric(cuisine_label, primary_cuisine["cuisine"])
    budget_column.metric("Maximum budget", f"₹{result['final_budget']:.0f}", "User limit")

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
        cuisine_source = "prediction" if prediction_source == "history_model" else "preference"
        st.info(
            f"No matching dishes were found for the top {cuisine_source}, "
            f"{primary_cuisine['cuisine']}. The workflow used the second {cuisine_source}, "
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
        first_step = (
            "The classifier predicted cuisine preference from behavioral order history."
            if prediction_source == "history_model"
            else "The selected onboarding cuisines handled the new-user cold start."
        )
        st.markdown(
            f"""
            1. {first_step}
            2. The user's explicit budget remained the hard price limit.
            3. Pandas selected real dishes matching city, locality, cuisine, price, and food preference.
            4. RAG retrieved common ingredients and allergens from the culinary knowledge base.
            5. Deterministic safety rules rejected known conflicts and marked uncertain recipes with warnings.
            6. Gemini explained only the validated candidates supplied by the workflow.
            """
        )
