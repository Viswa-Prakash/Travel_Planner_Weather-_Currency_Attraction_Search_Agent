# Travel Planner AI Agent

A conversational **Travel Planner agent** built using [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain), featuring:

- **Weather forecasts** for any city (real-time)
- **Live currency conversion rates**
- **Local restaurants and events** search
- **Math/calculations** (for planning budgets, conversions, etc.)
- **Step-by-step, tool-using reasoning** via the ReAct pattern

This agent provides a single, clear, actionable answer for each query, based on the latest data.

---

##  Features

- **LLM Orchestration:** Multi-step reasoning to combine query, tool calls, and final summarized advice
- **Multiple APIs**: OpenWeatherMap, AlphaVantage, Google Search/Serper, and a Python REPL
- **No “tool clutter”:** The user sees only the final, well-worded, actionable answer
- **Streamlit Web UI:** Easy to use, copy/paste your plans

---

##  Setup

### 1. Clone this repository

```bash
git clone https://github.com/Viswa-Prakash/Travel_Planner_Weather-_Currency_Attraction_Search_Agent.git
cd Travel_Planner_Weather-_Currency_Attraction_Search_Agent

### 2. Install dependencies
```bash
pip install -r requirements.txt


### 3. Set API keys
Create a .env file in your project root with:
```bash
OPENAI_API_KEY=sk-xxxxxxxxx
SERPER_API_KEY=serper_xxxxxx
GOOGLE_CSE_ID=xxxxxxx
GOOGLE_API_KEY=xxxxxxx
OPENWEATHERMAP_API_KEY=xxxxxxxx
ALPHA_VANTAGE_API_KEY=xxxxxxxx


### 4. Run the app
1. Start the app:

```bash
streamlit run app.py

2. Ask anything, like:
“We’re planning a trip to Tokyo in March. What’s the weather, how much is ¥10,000 in USD, and what are top events and vegan restaurants?”
“Where’s the best place for pizza in Naples and what’s the weather this weekend?”
“How do I convert 500 Euro to USD, and list 2 attractions in Paris?”

3. The agent responds with a single, professional summary.