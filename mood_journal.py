import gradio as gr
from transformers import pipeline
import pandas as pd
import datetime

# Initialize the sentiment/emotion analysis pipeline
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Create a DataFrame to store the journal entries
columns = ["Timestamp", "Entry", "Emotion"]
journal_df = pd.DataFrame(columns=columns)

# Process the journal entry and predict the emotion
def analyze_entry(entry):
    # Get sentiment analysis
    results = emotion_pipeline(entry)
    emotion = results[0]['label']

    # Log entry with timestamp and emotion
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[timestamp, entry, emotion]], columns=columns)
    global journal_df
    journal_df = pd.concat([journal_df, new_entry], ignore_index=True)

    # Display log and most recent emotion
    recent_log = journal_df.tail(5).to_dict(orient='records')
    return f"Your entry has been logged. Your entry was classified as: {emotion}", recent_log

# Create the Gradio interface
iface = gr.Interface(
    fn=analyze_entry,
    inputs="text",
    outputs=["text", "text"],
    title="Mood Journal Chatbot",
    description="Log your daily thoughts and analyze your mood.",
    live=True,
)

iface.launch()