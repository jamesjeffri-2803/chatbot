from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
from typing import Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from ibm_watson import NaturalLanguageUnderstandingV1, AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

app = FastAPI()

# --- Watson Credentials (replace with your actual credentials or use env vars) ---
WATSON_NLU_APIKEY = os.getenv('WATSON_NLU_APIKEY', 'your-nlu-apikey')
WATSON_NLU_URL = os.getenv('WATSON_NLU_URL', 'your-nlu-url')
WATSON_ASSISTANT_APIKEY = os.getenv('WATSON_ASSISTANT_APIKEY', 'your-assistant-apikey')
WATSON_ASSISTANT_URL = os.getenv('WATSON_ASSISTANT_URL', 'your-assistant-url')
WATSON_ASSISTANT_ID = os.getenv('WATSON_ASSISTANT_ID', 'your-assistant-id')

# --- Watson Clients ---
nlu_authenticator = IAMAuthenticator(WATSON_NLU_APIKEY)
nlu_client = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=nlu_authenticator
)
nlu_client.set_service_url(WATSON_NLU_URL)

assistant_authenticator = IAMAuthenticator(WATSON_ASSISTANT_APIKEY)
assistant_client = AssistantV2(
    version='2021-06-14',
    authenticator=assistant_authenticator
)
assistant_client.set_service_url(WATSON_ASSISTANT_URL)

# --- Training Data ---
training_sentences = [
    "How can I save more money?",
    "Best savings account?",
    "How do I reduce my taxable income?",
    "What are the tax benefits?",
    "Where should I invest?",
    "Best investment options?",
    "How can I earn money as a student?",
    "Ways for students to make money",
    "How to invest as a student?",
    "Where should students invest?",
    "How to earn money online?",
    "How to start a side hustle?",
    "How to make passive income?",
    "How to save money as a student?",
    "How to manage money as a student?",
]
training_labels = [
    "savings",
    "savings",
    "tax",
    "tax",
    "investment",
    "investment",
    "student_earn",
    "student_earn",
    "student_invest",
    "student_invest",
    "earn_online",
    "side_hustle",
    "passive_income",
    "student_savings",
    "student_savings",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)
clf = MultinomialNB()
clf.fit(X, training_labels)

class Profile(BaseModel):
    name: str
    age: int
    occupation: str
    monthly_income: float
    monthly_expenses: float
    financial_goal: str
    cost_items: str

class Query(BaseModel):
    question: str
    profile: Profile

# --- Utility Functions ---
def parse_costs(cost_items: str) -> Dict[str, float]:
    costs = {}
    for item in cost_items.split(","):
        if ":" in item:
            name, value = item.split(":")
            try:
                costs[name.strip()] = float(value.strip())
            except ValueError:
                continue
    return costs

def budget_summary(profile: Dict[str, Any], costs: Dict[str, float]) -> str:
    total_costs = sum(costs.values())
    surplus = profile['monthly_income'] - total_costs
    main_use = max(costs, key=costs.get) if costs else "N/A"
    summary = f"Budget Summary for {profile['name']}\n"
    summary += f"Monthly Income: ₹{profile['monthly_income']}\n"
    summary += f"Total Costs: ₹{total_costs}\n"
    summary += f"Main Use of Money: {main_use} (₹{costs.get(main_use,0)})\n"
    summary += f"Monthly Savings (after costs): ₹{surplus}\n"
    if surplus > 0:
        summary += "Great job! You are saving money each month."
    else:
        summary += "Warning: Your costs exceed your income. Consider reducing spending."
    return summary

# --- Watson Utility Functions ---
def watson_nlu_intent(text: str) -> str:
    try:
        response = nlu_client.analyze(
            text=text,
            features={
                'categories': {},
                'entities': {},
                'keywords': {},
                'sentiment': {},
                'emotion': {},
            }
        ).get_result()
        # Example: extract first keyword as intent (customize as needed)
        keywords = response.get('keywords', [])
        if keywords:
            return keywords[0]['text']
        return 'unknown'
    except Exception as e:
        return 'error'

def watson_assistant_response(text: str, session_id: str = None) -> str:
    try:
        if not session_id:
            session_id = assistant_client.create_session(
                assistant_id=WATSON_ASSISTANT_ID
            ).get_result()['session_id']
        response = assistant_client.message(
            assistant_id=WATSON_ASSISTANT_ID,
            session_id=session_id,
            input={
                'message_type': 'text',
                'text': text
            }
        ).get_result()
        # Example: extract first generic response
        output = response.get('output', {})
        generic = output.get('generic', [])
        if generic and 'text' in generic[0]:
            return generic[0]['text']
        return 'No response.'
    except Exception as e:
        return f'Error: {str(e)}'

def construct_response(status: str, data: dict = None, message: str = "") -> dict:
    return {
        "status": status,
        "data": data if data else {},
        "message": message
    }

# --- Endpoints ---
@app.post("/chat")
def chat(query: Query):
    user_input = query.question
    profile = query.profile.dict()
    # Try Watson Assistant first
    watson_reply = watson_assistant_response(user_input)
    if watson_reply and not watson_reply.startswith('Error'):
        return construct_response("success", {"response": watson_reply, "intent": "watson"})
    # Fallback to local ML
    X_test = vectorizer.transform([user_input])
    intent = clf.predict(X_test)[0]
    # ...existing ML response logic...
    if intent == "savings":
        advice = f"💰 Hi {profile['name']}, as a {profile['occupation']}, consider setting a budget, tracking spending, and automating deposits."
        if profile['monthly_income'] > profile['monthly_expenses']:
            advice += " You have a positive cash flow. Consider putting the surplus into a high-yield savings account."
        else:
            advice += " Try to reduce expenses to increase your savings rate."
        return construct_response("success", {"response": advice, "intent": intent})
    elif intent == "tax":
        return construct_response("success", {"response": f"📑 {profile['name']}, keep records, explore deductions, and consider consulting a tax advisor. Tax tips can vary for {profile['occupation'].lower()}s.", "intent": intent})
    elif intent == "investment":
        return construct_response("success", {"response": f"📈 {profile['name']}, diversify your portfolio, understand your risk tolerance, and review regularly. As a {profile['occupation']}, consider index funds or retirement accounts.", "intent": intent})
    elif intent == "student_earn":
        return construct_response("success", {"response": "🧑‍🎓 As a student, you can earn money through part-time jobs, freelancing, tutoring, internships, or online gigs like content creation and surveys.", "intent": intent})
    elif intent == "student_invest":
        return construct_response("success", {"response": "📚 Students can start investing with small amounts in mutual funds, index funds, or fractional shares using beginner-friendly platforms. Focus on learning and long-term growth.", "intent": intent})
    elif intent == "earn_online":
        return construct_response("success", {"response": "🌐 You can earn money online by freelancing, selling products, teaching, or creating digital content (YouTube, blogging, etc.).", "intent": intent})
    elif intent == "side_hustle":
        return construct_response("success", {"response": "💼 Side hustles include freelancing, ride-sharing, delivery services, online tutoring, and selling crafts or digital products.", "intent": intent})
    elif intent == "passive_income":
        return construct_response("success", {"response": "💸 Passive income ideas: invest in stocks/dividends, real estate, create digital products, or peer-to-peer lending.", "intent": intent})
    elif intent == "student_savings":
        return construct_response("success", {"response": "🧑‍🎓 As a student, save money by budgeting, using student discounts, buying used textbooks, and cooking at home. Track your expenses regularly.", "intent": intent})
    else:
        return construct_response("error", {}, "🤖 I can help with savings, taxes, investments, earning, and more. Please ask again.")

@app.post("/budget")
def budget(profile: Profile):
    costs = parse_costs(profile.cost_items)
    summary = budget_summary(profile.dict(), costs)
    return construct_response("success", {"summary": summary})

@app.post("/nlu")
def nlu(query: Query):
    user_input = query.question
    # Try Watson NLU first
    intent = watson_nlu_intent(user_input)
    if intent not in ['error', 'unknown']:
        return construct_response("success", {"intent": intent})
    # Fallback to local ML
    X_test = vectorizer.transform([user_input])
    intent = clf.predict(X_test)[0]
    return construct_response("success", {"intent": intent})

@app.post("/spending")
def spending(profile: Profile):
    costs = parse_costs(profile.cost_items)
    if costs:
        main_use = max(costs, key=costs.get)
        return construct_response("success", {
            "largest_expense": main_use,
            "amount": costs[main_use],
            "all_expenses": costs,
            "total_costs": sum(costs.values()),
            "monthly_savings": profile.monthly_income - sum(costs.values())
        })
    else:
        return construct_response("error", {}, "No cost data entered.")

@app.post("/generate")
def generate(query: Query):
    user_input = query.question
    # Use Watson Assistant for generation
    watson_reply = watson_assistant_response(user_input)
    if watson_reply and not watson_reply.startswith('Error'):
        return construct_response("success", {"generated": watson_reply})
    # Fallback to echo
    return construct_response("success", {"generated": f"Generated response for: {user_input}"})