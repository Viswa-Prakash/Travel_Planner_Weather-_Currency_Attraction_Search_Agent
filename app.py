import streamlit as st
from langchain_core.messages import HumanMessage
from ReAct_Agent import travel_agent

st.set_page_config(page_title="Travel Planner", page_icon=":earth_americas:")

st.title("Travel Planner with Weather, Currency, and Attraction Search")

st.markdown("""
            "We’re planning a Tokyo trip in March. 
            What’s the typical weather? 
            How much is ¥10,000 in USD now? 
            List two popular events and nearby vegan restaurants."
""")

with st.form("travel_form"):
    user_query = st.text_area("Enter your travel plan here:", height=60)
    submitted = st.form_submit_button("Ask Agent")

if submitted and user_query.strip():
    with st.spinner("Agent is analyzing..."):
        output = travel_agent.invoke({"messages": [HumanMessage(content=user_query)]})
        # Show **only the last agent message** (the Final Answer)
        final_message = None
        for msg in reversed(output["messages"]):
            content = getattr(msg, "content", "")
            if "final answer" in content.lower():
                final_message = content
                break
        if not final_message:
            # fallback: just show last assistant/system message
            last = output["messages"][-1]
            final_message = getattr(last, "content", str(last))
        st.markdown("**Here’s a clear summary of your requests and answers:**\n\n" + final_message)
