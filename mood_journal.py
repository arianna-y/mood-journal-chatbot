import gradio as gr
from transformers import pipeline
import pandas as pd
import datetime
import random
import os
import matplotlib.pyplot as plt

# Initialize the sentiment/emotion analysis pipeline
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Create a DataFrame to store the journal entries
columns = ["Timestamp", "Entry", "Emotion"]
journal_df = pd.DataFrame(columns=columns)

# Random prompts
prompts = [
    "What made you smile today?",
    "Describe a moment that made you feel proud.",
    "What is something you're looking forward to?",
    "How did you overcome a challenge today?",
    "What is one thing you learned today?",
    "is there anything that's been on your mind lately?",
    "Describe your day using a weather metaphor (sunny, cloudy, stormy).",
    "What is something you're grateful for today?"
]

csv_file = "journal_entries.csv"
if os.path.exists(csv_file):
    journal_df = pd.read_csv(csv_file)
else:
    journal_df = pd.DataFrame(columns=["timestamp", "entry", "emotion"])

# Process the journal entry and predict the emotion
def analyze_entry(entry):
    if not entry.strip():
        return "Please enter a valid journal entry.", None
    
    # Get sentiment analysis
    results = emotion_pipeline(entry)
    emotion = results[0]['label']

    # Log entry with timestamp and emotion
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[timestamp, entry, emotion]], columns=["Timestamp", "Entry", "Emotion"])
    global journal_df
    journal_df = pd.concat([journal_df, new_entry], ignore_index=True)

    # Display log and most recent emotion
    recent_log = journal_df.tail(5)[["Timestamp", "Entry", "Emotion"]]
    return f"Your entry has been logged. Your entry was classified as: {emotion}", recent_log, plot_mood_graph(journal_df)

# Plot mood graph
def plot_mood_graph(log):
    mood_counts = log["Emotion"].value_counts()
    fig, ax = plt.subplots()
    mood_counts.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Mood Distribution")
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Count")
    return fig

# Prompt generator
def get_random_prompt():
    return random.choice(prompts)

# Create the Gradio interface
# iface = gr.Interface(
#     fn=analyze_entry,
#     inputs="text",
#     outputs=["text", "text"],
#     title="Mood Journal Chatbot",
#     description="Log your daily thoughts and analyze your mood.",
#     live=True,
# )
with gr.Blocks() as iface:
    gr.Markdown("## Mood Journal Chatbot")
    gr.Markdown("Log your daily thoughts and analyze your mood.")

    with gr.Row():
        entry = gr.Textbox(lines=5, placeholder="Write your thoughts here...", label="Journal Entry")
        prompt_btn = gr.Button("Get Random Prompt")

    prompt_display = gr.Textbox(label="Random Prompt", interactive=False)

    submit_btn = gr.Button("Submit Entry")
    mood_output = gr.Textbox(label="Mood Analysis")
    log_output = gr.Dataframe(headers=["Timestamp", "Entry", "Mood"], label="Recent Entries", interactive=False, show_label=True)
    mood_plot = gr.Plot(label="Mood Distribution")

    prompt_btn.click(get_random_prompt, outputs=prompt_display)
    submit_btn.click(analyze_entry, inputs=entry, outputs=[mood_output, log_output, mood_plot])

iface.launch()