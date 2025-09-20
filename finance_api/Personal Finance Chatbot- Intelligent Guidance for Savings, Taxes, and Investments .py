import streamlit as st
import requests
import json

st.set_page_config(page_title="Personal Finance Chatbot", page_icon="💰")
pages = ["NLU Analysis", "Q & A", "Budget Summary", "Spending Insights"]
page = st.sidebar.radio("Go to", pages)

API_BASE_URL = "http://127.0.0.1:8000"

API_ROUTES = {
    "nlu": f"{API_BASE_URL}/nlu",
    "generate": f"{API_BASE_URL}/generate",
    "budget": f"{API_BASE_URL}/budget",
    "summary": f"{API_BASE_URL}/summary",
    "spending_insights": f"{API_BASE_URL}/spending"
}

# --- User Profile Form ---
def user_profile_form():
    st.header("User Profile")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=10, max_value=100, value=18)
    occupation = st.selectbox("Occupation", ["Student", "Professional", "Retired", "Other"])
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=2000)
    monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=1000)
    financial_goal = st.text_input("Financial Goal")
    cost_items = st.text_area("List your monthly costs (e.g. Rent:5000, Food:2000, Transport:1000)")
    return {
        "name": name,
        "age": age,
        "occupation": occupation,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "financial_goal": financial_goal,
        "cost_items": cost_items,
    }

profile = user_profile_form()

# --- Main Chat UI ---
if page == "NLU Analysis":
    st.title("NLU Analysis")
    st.write("This page will show Natural Language Understanding analysis of your queries.")
    user_nlu_input = st.text_input("Enter a financial question for NLU analysis:")
    if user_nlu_input:
        payload = {"question": user_nlu_input, "profile": profile}
        response = requests.post(API_ROUTES["nlu"], json=payload)
        resp_json = response.json()
        if resp_json["status"] == "success":
            intent = resp_json["data"].get("intent", "")
            st.markdown(f"**Detected Intent:** {intent}")
        else:
            st.error(resp_json.get("message", "NLU error"))
        st.markdown(f"**Profile:** {profile}")
        st.markdown(f"**Raw Input:** {user_nlu_input}")
        with st.expander("Show raw JSON response"):
            st.json(resp_json)

elif page == "Q & A":
    st.title("Finance Q & A Chatbot")
    st.write("Ask any question about savings, taxes, investments, or student finance.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("Ask me a question:", "", key="qa_input")
    if st.button("Send", key="qa_send"):
        if user_input:
            payload = {"question": user_input, "profile": profile}
            response = requests.post(f"{API_BASE_URL}/chat", json=payload)
            resp_json = response.json()
            if resp_json["status"] == "success":
                bot_response = resp_json["data"].get("response", "")
            else:
                bot_response = resp_json.get("message", "Chat error")
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", bot_response))
            st.session_state.last_chat_json = resp_json
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")
    if "last_chat_json" in st.session_state:
        with st.expander("Show raw JSON response"):
            st.json(st.session_state.last_chat_json)

elif page == "Budget Summary":
    st.title("Budget Summary")
    response = requests.post(f"{API_BASE_URL}/budget", json=profile)
    resp_json = response.json()
    if resp_json["status"] == "success":
        summary = resp_json["data"].get("summary", "")
        st.markdown(summary)
    else:
        st.error(resp_json.get("message", "Budget error"))
    with st.expander("Show raw JSON response"):
        st.json(resp_json)

elif page == "Spending Insights":
    st.title("Spending Insights")
    st.write("Here are your spending insights based on your monthly costs:")
    response = requests.post(f"{API_BASE_URL}/spending", json=profile)
    resp_json = response.json()
    if resp_json["status"] == "success":
        data = resp_json["data"]
        st.markdown(f"**Largest Expense:** {data['largest_expense']} (₹{data['amount']})")
        st.markdown(f"**All Expenses:**")
        for k, v in data["all_expenses"].items():
            st.markdown(f"- {k}: ₹{v}")
        st.markdown(f"**Total Costs:** ₹{data['total_costs']}")
        st.markdown(f"**Monthly Savings:** ₹{data['monthly_savings']}")
        st.info("Consider reducing your largest expense or finding ways to optimize your spending.")
    else:
        st.warning(resp_json.get("message", "Spending error"))
    with st.expander("Show raw JSON response"):
        st.json(resp_json)
