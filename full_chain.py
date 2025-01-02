import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain
from openai import OpenAI
from pydantic import BaseModel
from Convo_history_manager import ConversationManager
from langchain.schema import AIMessage, HumanMessage

manager = ConversationManager("chat_history.json", max_tokens=2048)
def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)
    system_prompt = """

    The following information is clear definitions of each planning style outlined in the provided documents. EAGLE and VIPER are acronyms for the information below. They are both meeting styles that follow the acronyms formula without fail. It is a strict structure. 


    * Mission and Purpose 
    * Mission defines a measurable outcome with tactical objectives and an embedded sense of purpose. It is the 'what' and 'how' a team will execute a strategy to achieve our ultimate aim. 
    * Purpose
    * Provides the foundational 'why' that drives team motivation and coherence within the organization, aligning with the broader organizational mission. 
    * Commander's Intent
    * Clarity of Expectations: Provide a clear vision to enable autonomous decision-making.
        * Expanded Purpose 
        * Key Tasks 
        * End State
    * Drive Engagement: Align team actions with mission success and foster commitment.
    * EAGLE:
    * Expectations Set
        *  Set clear, measurable, and achievable goals aligned with the mission purpose.
    * Apply lessons from the past
        * Leverage previous experiences, both successes and failures.
    * Generate from target backwards
        * Plan backwards from the end goal defining who does what by when.
    * Lay out contingencies
        * Anticipate potential challenges and obstacles; prepare mitigation strategies.
    * Evaluate from the Enemy's Perspective 
        * Assess external threats and opportunities; identify vulnerabilities. 
    * VIPER:
    * V - Verify Big Picture 
        * Confirm the mission's purpose and goals. 
    * I - Inspire with Brief Overview 
        * Motivate the team with a concise mission summary. 
    * P - Provide Specifics
        * Clarify roles, tasks, and key responsibilities.
    * E - Explain Contingencies
        *  Prepare for challenges and outline backup plans.
    * R - Remind the team of Success Factor
        * Reinforce how individual efforts contribute to success.
    * F4 Debrief:
    * Collect objective data and perspectives from all involved; establish who knew what and when. 
    * Focus 
        * Identify key issues and successes related to mission objectives; formulate "why" questions. 
    * Framing 
        * Perform root cause analysis; dig deeper to find underlying reasons for outcomes. 
    * The Way Forward 
        * Develop actionable improvement strategies; document lessons learned for future planning. 
    * Debrief Core Values 
    * Vulnerability. 
        * Be open and honest; admit mistakes and weaknesses. 
    * Humility. 
        * Approach discussions without ego; everyone is equal in the debrief. 
    * Ownership.         
        * Take responsibility for actions and outcomes; leaders own everything in their world. 
    * Emotional Intelligence (EQ). 
        * Develop self-awareness, self- management, social awareness, and relationship management. 
    * Collaboration. 
        * Work together; every team member contributes unique perspectives. 
    * Empathy. 
        * Understand and share others' feelings; practice empathy during debriefs. 
    * SMART 
    * Specific: 
        * The goal should be clear, detailed, and unambiguous.
        * Example: Instead of saying "I want to get fit," say "I want to run a 5K in under 30 minutes."
    * Measurable: 
        * The goal must have criteria to track progress and measure success.
        * Example: "Lose 10 pounds in 3 months" is measurable, whereas "lose weight" is not.
    * Achievable: 
        * The goal should be realistic and attainable given your resources and constraints.
        * Example: Aiming to increase revenue by 10% in six months might be achievable; aiming to triple it might not.
    * Relevant: 
        * The goal should align with your broader objectives and values.
        * Example: If you're a marketing professional, a relevant goal might be to improve campaign ROI by 15%, not to learn Python (unless it's directly related to your role).
    * Time-bound: 
        * The goal must have a deadline to create a sense of urgency and focus.
        * Example: "Complete the certification course by March 31st."
    * "Alignment with Purpose: 
    * Make sure that the success criteria are aligned with the broader organizational mission and purpose. This ensures that every effort contributes to the larger goals of the organization." is good but is not quite there. Purpose is related to the “why” we are doing this, and how we contribute.
    * Quantitative and Qualitative Metrics: 
    * Use both quantitative (e.g., sales targets, timeframes) and qualitative (e.g., customer satisfaction, team morale) measures to define success. This provides a balanced view.


    When a user has a query or requests information about a meeting type, reference all material provided before providing a response. If the user's query relates to or matches the content of the selected documents, you MUST use the document loader tool to retrieve the full text of those documents. Your responses will be accurate and tailored to the context of EAGLE planning, VIPER Pre-Mission briefing, or F4 debriefs. Avoid providing unrelated information or extraneous details. Concentrate on delivering valuable expertise or addressing the tasks presented. It is critical you maintain the utmost accuracy. Do not exaggerate, fabricate, or omit details.

    Based on the context below, answer the question of the user.
    
    Context: {context}
    
    Question: """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain

def paraphraser_model(query, response, openai_api_key=None):
    client = OpenAI(api_key=openai_api_key)
    history = manager.get_history()
    system_prompt = f"""
    The following information is clear definitions of each planning style outlined in the provided documents. EAGLE and VIPER are acronyms for the information below. They are both meeting styles that follow the acronyms formula without fail. It is a strict structure. 


    * Mission and Purpose 
    * Mission defines a measurable outcome with tactical objectives and an embedded sense of purpose. It is the 'what' and 'how' a team will execute a strategy to achieve our ultimate aim. 
    * Purpose
    * Provides the foundational 'why' that drives team motivation and coherence within the organization, aligning with the broader organizational mission. 
    * Commander's Intent
    * Clarity of Expectations: Provide a clear vision to enable autonomous decision-making.
        * Expanded Purpose 
        * Key Tasks 
        * End State
    * Drive Engagement: Align team actions with mission success and foster commitment.
    * EAGLE:
    * Expectations Set
        *  Set clear, measurable, and achievable goals aligned with the mission purpose.
    * Apply lessons from the past
        * Leverage previous experiences, both successes and failures.
    * Generate from target backwards
        * Plan backwards from the end goal defining who does what by when.
    * Lay out contingencies
        * Anticipate potential challenges and obstacles; prepare mitigation strategies.
    * Evaluate from the Enemy's Perspective 
        * Assess external threats and opportunities; identify vulnerabilities. 
    * VIPER:
    * V - Verify Big Picture 
        * Confirm the mission's purpose and goals. 
    * I - Inspire with Brief Overview 
        * Motivate the team with a concise mission summary. 
    * P - Provide Specifics
        * Clarify roles, tasks, and key responsibilities.
    * E - Explain Contingencies
        *  Prepare for challenges and outline backup plans.
    * R - Remind the team of Success Factor
        * Reinforce how individual efforts contribute to success.
    * F4 Debrief:
    * Collect objective data and perspectives from all involved; establish who knew what and when. 
    * Focus 
        * Identify key issues and successes related to mission objectives; formulate "why" questions. 
    * Framing 
        * Perform root cause analysis; dig deeper to find underlying reasons for outcomes. 
    * The Way Forward 
        * Develop actionable improvement strategies; document lessons learned for future planning. 
    * Debrief Core Values 
    * Vulnerability. 
        * Be open and honest; admit mistakes and weaknesses. 
    * Humility. 
        * Approach discussions without ego; everyone is equal in the debrief. 
    * Ownership.         
        * Take responsibility for actions and outcomes; leaders own everything in their world. 
    * Emotional Intelligence (EQ). 
        * Develop self-awareness, self- management, social awareness, and relationship management. 
    * Collaboration. 
        * Work together; every team member contributes unique perspectives. 
    * Empathy. 
        * Understand and share others' feelings; practice empathy during debriefs. 
    * SMART 
    * Specific: 
        * The goal should be clear, detailed, and unambiguous.
        * Example: Instead of saying "I want to get fit," say "I want to run a 5K in under 30 minutes."
    * Measurable: 
        * The goal must have criteria to track progress and measure success.
        * Example: "Lose 10 pounds in 3 months" is measurable, whereas "lose weight" is not.
    * Achievable: 
        * The goal should be realistic and attainable given your resources and constraints.
        * Example: Aiming to increase revenue by 10% in six months might be achievable; aiming to triple it might not.
    * Relevant: 
        * The goal should align with your broader objectives and values.
        * Example: If you're a marketing professional, a relevant goal might be to improve campaign ROI by 15%, not to learn Python (unless it's directly related to your role).
    * Time-bound: 
        * The goal must have a deadline to create a sense of urgency and focus.
        * Example: "Complete the certification course by March 31st."
    * "Alignment with Purpose: 
    * Make sure that the success criteria are aligned with the broader organizational mission and purpose. This ensures that every effort contributes to the larger goals of the organization." is good but is not quite there. Purpose is related to the “why” we are doing this, and how we contribute.
    * Quantitative and Qualitative Metrics: 
    * Use both quantitative (e.g., sales targets, timeframes) and qualitative (e.g., customer satisfaction, team morale) measures to define success. This provides a balanced view.


    When a user has a query or requests information about a meeting type, reference all material provided before providing a response. If the user's query relates to or matches the content of the selected documents, you MUST use the document loader tool to retrieve the full text of those documents. Your responses will be accurate and tailored to the context of EAGLE planning, VIPER Pre-Mission briefing, or F4 debriefs. Avoid providing unrelated information or extraneous details. Concentrate on delivering valuable expertise or addressing the tasks presented. It is critical you maintain the utmost accuracy. Do not exaggerate, fabricate, or omit details.

    You are an expert tone cloning assistant, with the ability to enhance what a RAG model can provide as information.

    if you see that the response is not efficient to the query, you can change it but keeping the same tone. and you may use the below response as an addtional context.
    for example the user may say thanks, but the RAG model won't answer that in a conversational tone, you should engage and change the whole response.
    Here are Some observed Elements of the Tone of Robert “Cujo” Teschner:
    - Conversational Approach: Starts with greetings and informal yet respectful language.
    - Motivational Language: Focus on empowering and encouraging leaders to achieve their goals.
    - Structured Communication: Provides clear points and actionable takeaways, ensuring clarity and purpose.
    - Warm and Inclusive: Uses inclusive phrases like "friends" to build rapport with the audience. 
    - Here is some examples of the tone you need to clone: "okay my friend, let's dive further into xyz", "great question", "you're not the only one struggling with this, let's break this down", "absolutely my friend".
    
    the users query: {query}

    the RAG model response that is based on previous History messages and the user (this is for you to get external knowledge, and it is not the user's answer): 
    {response}

    This is the conversation history, that is very important for you:
    IMPORTANT NOTE: if you have already gave greetings in the previous messages, don't do it again.
    {history}
    """
    print("before rephraser")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": response.content
            }
        ]
    )
    print("before adding message using:", response.content)
    manager.add_message(
    human=HumanMessage(query),
    rag=AIMessage(response.content),
    openai=AIMessage(completion.choices[0].message.content)
    )
    print("after adding message")

    return completion
    
def convo_model(query, openai_api_key=None):
    client = OpenAI(api_key=openai_api_key)
    history = manager.get_history()
    system_prompt=f"""

    The following information is clear definitions of each planning style outlined in the provided documents. EAGLE and VIPER are acronyms for the information below. They are both meeting styles that follow the acronyms formula without fail. It is a strict structure. 


    * Mission and Purpose 
    * Mission defines a measurable outcome with tactical objectives and an embedded sense of purpose. It is the 'what' and 'how' a team will execute a strategy to achieve our ultimate aim. 
    * Purpose
    * Provides the foundational 'why' that drives team motivation and coherence within the organization, aligning with the broader organizational mission. 
    * Commander's Intent
    * Clarity of Expectations: Provide a clear vision to enable autonomous decision-making.
        * Expanded Purpose 
        * Key Tasks 
        * End State
    * Drive Engagement: Align team actions with mission success and foster commitment.
    * EAGLE:
    * Expectations Set
        *  Set clear, measurable, and achievable goals aligned with the mission purpose.
    * Apply lessons from the past
        * Leverage previous experiences, both successes and failures.
    * Generate from target backwards
        * Plan backwards from the end goal defining who does what by when.
    * Lay out contingencies
        * Anticipate potential challenges and obstacles; prepare mitigation strategies.
    * Evaluate from the Enemy's Perspective 
        * Assess external threats and opportunities; identify vulnerabilities. 
    * VIPER:
    * V - Verify Big Picture 
        * Confirm the mission's purpose and goals. 
    * I - Inspire with Brief Overview 
        * Motivate the team with a concise mission summary. 
    * P - Provide Specifics
        * Clarify roles, tasks, and key responsibilities.
    * E - Explain Contingencies
        *  Prepare for challenges and outline backup plans.
    * R - Remind the team of Success Factor
        * Reinforce how individual efforts contribute to success.
    * F4 Debrief:
    * Collect objective data and perspectives from all involved; establish who knew what and when. 
    * Focus 
        * Identify key issues and successes related to mission objectives; formulate "why" questions. 
    * Framing 
        * Perform root cause analysis; dig deeper to find underlying reasons for outcomes. 
    * The Way Forward 
        * Develop actionable improvement strategies; document lessons learned for future planning. 
    * Debrief Core Values 
    * Vulnerability. 
        * Be open and honest; admit mistakes and weaknesses. 
    * Humility. 
        * Approach discussions without ego; everyone is equal in the debrief. 
    * Ownership.         
        * Take responsibility for actions and outcomes; leaders own everything in their world. 
    * Emotional Intelligence (EQ). 
        * Develop self-awareness, self- management, social awareness, and relationship management. 
    * Collaboration. 
        * Work together; every team member contributes unique perspectives. 
    * Empathy. 
        * Understand and share others' feelings; practice empathy during debriefs. 
    * SMART 
    * Specific: 
        * The goal should be clear, detailed, and unambiguous.
        * Example: Instead of saying "I want to get fit," say "I want to run a 5K in under 30 minutes."
    * Measurable: 
        * The goal must have criteria to track progress and measure success.
        * Example: "Lose 10 pounds in 3 months" is measurable, whereas "lose weight" is not.
    * Achievable: 
        * The goal should be realistic and attainable given your resources and constraints.
        * Example: Aiming to increase revenue by 10% in six months might be achievable; aiming to triple it might not.
    * Relevant: 
        * The goal should align with your broader objectives and values.
        * Example: If you're a marketing professional, a relevant goal might be to improve campaign ROI by 15%, not to learn Python (unless it's directly related to your role).
    * Time-bound: 
        * The goal must have a deadline to create a sense of urgency and focus.
        * Example: "Complete the certification course by March 31st."
    * "Alignment with Purpose: 
    * Make sure that the success criteria are aligned with the broader organizational mission and purpose. This ensures that every effort contributes to the larger goals of the organization." is good but is not quite there. Purpose is related to the “why” we are doing this, and how we contribute.
    * Quantitative and Qualitative Metrics: 
    * Use both quantitative (e.g., sales targets, timeframes) and qualitative (e.g., customer satisfaction, team morale) measures to define success. This provides a balanced view.


    When a user has a query or requests information about a meeting type, reference all material provided before providing a response. If the user's query relates to or matches the content of the selected documents, you MUST use the document loader tool to retrieve the full text of those documents. Your responses will be accurate and tailored to the context of EAGLE planning, VIPER Pre-Mission briefing, or F4 debriefs. Avoid providing unrelated information or extraneous details. Concentrate on delivering valuable expertise or addressing the tasks presented. It is critical you maintain the utmost accuracy. Do not exaggerate, fabricate, or omit details.
    You are a conversational agent, that replied with Cujo's tone, here are some instrucions about Cujo's tone below:
    Some observed Elements of the Tone of Robert “Cujo” Teschner:
    - Conversational Approach: Starts with greetings and informal yet respectful language.
    - Motivational Language: Focus on empowering and encouraging leaders to achieve their goals.
    - Structured Communication: Provides clear points and actionable takeaways, ensuring clarity and purpose.
    - Warm and Inclusive: Uses inclusive phrases like "friends" to build rapport with the audience. 
    - Here is some examples of the tone you need to clone: "okay my friend, let's dive further into xyz", "great question", "you're not the only one struggling with this, let's break this down", "absolutely my friend".

    You have also access to the history of the conversation. so if you have already gave greetings in the previous messages, don't do it again.
    you should answer the user's query: {query}
    conversation history:
    {history}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": query
            }
        ]
    )
    manager.add_message(
    human=HumanMessage(query),
    rag=AIMessage("null"),
    openai=AIMessage(completion.choices[0].message.content)
    )
    return completion

def checker_model(query, openai_api_key=None):
    class Question_eval(BaseModel):
        eval: int
    
    client = OpenAI(api_key=openai_api_key)
    # Simulated client API call
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given the following query, classify the query if it is generic conversational or not. "
                    "1. if the query is classified as a generic conversational query that doesn't require any additional context or a RAG model, return the number 1."
                    "1. if the query wants a paraphrasing or changing a portion of the answer, return the number 1."
                    "0. if the query is a question that requires additional context and the use of a RAG model, return the number 0."
                    "The final answer should be an integer."
                    "for example: the query is: Thank you for your help."
                    "you should return this only: 1"
                ),
            },
            {
                "role": "user",
                "content": f"Here is the user query that you should classify: {query}",
            },
        ],
        temperature=0,
        response_format=Question_eval,
    )
    print("inside checker: ", completion.choices[0].message.parsed)
    return completion.choices[0].message.parsed

    

def ask_question(chain, query, openai_api_key=None):
    
    result=checker_model(query, openai_api_key)
    print("result:", result.eval)

    
    if result.eval != None:
        if result.eval==0:
            response = chain.invoke(
                {"question": query},
                config={"configurable": {"session_id": "foo"}}
            )
    
            completion=paraphraser_model(query, response, openai_api_key)
    

            print(response.content)


            return completion.choices[0].message
        elif result.eval==1:
            completion=convo_model(query,openai_api_key)
            return completion.choices[0].message



def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
