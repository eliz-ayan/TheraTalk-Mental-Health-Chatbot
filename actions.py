from typing import Any, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class ActionSetBreathingSlot(Action):
    def name(self) -> str:
        return "action_set_breathing_slot"

    def run(
        self, dispatcher, tracker, domain
    ) -> List[EventType]:
        return [SlotSet("is_in_breathing", True)]

class ActionSetCrisisSlot(Action):
    def name(self) -> str:
        return "action_set_crisis_slot"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:
        return [SlotSet("is_in_crisis", True)]

class ActionResetCrisisSlot(Action):
    def name(self) -> str:
        return "action_reset_crisis_slot"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:
        return [SlotSet("is_in_crisis", False)]

class ActionOpenAIRag(Action):
    def name(self) -> str:
        return "action_openai_rag"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:

        user_msg = tracker.latest_message.get("text")

      
        crisis_keywords = [
            "kill myself", "suicide", "end it all", "want to die", "hurt myself"
        ]
        if any(phrase in user_msg.lower() for phrase in crisis_keywords):
            dispatcher.utter_message(
                text="It sounds like you’re going through something very difficult. You're not alone. Would you like me to share some emergency resources or someone you can talk to right away?"
            )
            return [SlotSet("is_in_crisis", True)]

      
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.load_local(
            "cbt_index", embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

      

        prompt_template = PromptTemplate.from_template(

            """
        You are TheraTalk — a compassionate and skilled CBT therapist chatbot. 
        Your mission is to gently support users in exploring their thoughts and feelings using evidence-based Cognitive Behavioral Therapy (CBT) techniques.

        Your therapeutic tools include:
        - Cognitive restructuring (identifying and reframing unhelpful thoughts)
        - Behavioral activation (encouraging meaningful actions)
        - Problem-solving (collaborative exploration of next steps)

        When responding:
        - Reflect empathically on the user's message
        - Use Socratic questioning to invite curiosity and insight
        - Avoid sounding scripted or robotic — speak like a warm, caring human
        - Do NOT diagnose, give medical advice, or assume anything not mentioned
        - ONLY offer crisis support if the user clearly expresses intent to harm themselves or others
        - Use the context below ONLY as background to guide your tone and suggestions (do not reference it directly)

        ---

        Background context (for tone and therapeutic direction only):
        {context}

        User: {question}

        TheraTalk:
        """
        )


        llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", temperature=0.7, api_key=api_key
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False,
        )

        try:
            result = qa_chain.invoke({"query": user_msg})
            dispatcher.utter_message(text=result["result"])
        except Exception as e:
            dispatcher.utter_message(
                text="Sorry, I had trouble processing your message."
            )
            print(f"[Action Error] {str(e)}")

        return []

class ActionOpenAICognitiveRestructuring(Action):
    def name(self):
        return "action_openai_cognitive_restructuring"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):

        user_msg = tracker.latest_message.get("text")

        crisis_keywords = ["kill myself", "suicide", "end it all", "want to die", "hurt myself"]
        if any(phrase in user_msg.lower() for phrase in crisis_keywords):
            dispatcher.utter_message(
                text="It sounds like you’re going through something very difficult. You're not alone. Would you like me to share some emergency resources or someone you can talk to right away?")
            return []
      
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.load_local("cbt_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
      
        prompt_template = PromptTemplate.from_template("""
        You are a compassionate CBT therapist chatbot named TheraTalk, specializing in Cognitive Restructuring for anxiety.

        Your goal is to:
        - Reflect empathetically on what the user says
        - Help the user recognise and challenge unhelpful anxious thoughts
        - Use Socratic questioning to help the user shift perspective and reframe anxiety-inducing thoughts
        - Help the user gain a clearer sense of control over their emotions and anxiety

        ----                                                    

        Background context (for tone and therapeutic direction and Cogntitve restructing only):
                {context}

                User: {question}

                TheraTalk:
        """)

        llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7, api_key=api_key)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        try:
            result = qa_chain.invoke({"query": user_msg})
            dispatcher.utter_message(text=result["result"])
        except Exception as e:
            dispatcher.utter_message(text="Sorry, I had trouble processing your message.")
            print(f"[Action Error] {str(e)}")

        return []

class ActionOpenAIBehaviouralActivation(Action):
    def name(self):
        return "action_openai_behavioural_activation"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):

        user_msg = tracker.latest_message.get("text")

        crisis_keywords = ["kill myself", "suicide", "end it all", "want to die", "hurt myself"]
        if any(phrase in user_msg.lower() for phrase in crisis_keywords):
            dispatcher.utter_message(
                text="It sounds like you’re going through something very difficult. You're not alone. Would you like me to share some emergency resources or someone you can talk to right away?")
            return []

       
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.load_local("cbt_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

       
        prompt_template = PromptTemplate.from_template("""
        You are a compassionate CBT therapist chatbot named TheraTalk, specialising in Behavioral Activation to treat depression.

        Your goal is to:
        - Reflect empathetically on what the user says
        - Help the user identify patterns of avoidance or inaction related to depression
        - Encourage small, meaningful activities to improve mood and engagement
        - Help the user re-engage with positive behaviors and values, even in the face of depression

                ----                                                    

                Background context (for tone and therapeutic direction and Behvioural Activation techniques only):
                        {context}

                        User: {question}

                        TheraTalk:
        """)

        llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7, api_key=api_key)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )

        try:
            result = qa_chain.invoke({"query": user_msg})
            dispatcher.utter_message(text=result["result"])
        except Exception as e:
            dispatcher.utter_message(text="Sorry, I had trouble processing your message.")
            print(f"[Action Error] {str(e)}")

        return []

class ActionOpenAIContinueIntervention(Action):
    def name(self) -> str:
        return "action_openai_continue_intervention"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]) -> List[EventType]:
        
        user_message = tracker.latest_message.get('text')

        llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=api_key)

        
        prompt_template = PromptTemplate.from_template(
            """
            You are a compassionate mental health chatbot. Continue the conversation based on the following input from the user.

            User's message: {message}

            Continue offering support by providing Cognitive Behavioral Therapy techniques, empathy, or suggesting appropriate actions (e.g., somatic breathing, emotional check-ins). If the user is in a crisis, prioritize emergency support.
            """
        )

        prompt = prompt_template.format(message=user_message)

       
        try:
            response = llm(prompt)
            dispatcher.utter_message(response['choices'][0]['text'])
        except Exception as e:
            dispatcher.utter_message(text="Sorry, I encountered an issue while processing your request.")
            print(f"[Action Error] {str(e)}")

   
        return []

