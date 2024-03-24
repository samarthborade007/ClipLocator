import streamlit as st
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def extract_transcript_with_timestamps(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_with_timestamps = ''
        for segment in transcript_list:
            start_time = segment['start']
            end_time = segment['start'] + segment['duration']
            transcript_with_timestamps += f"[{format_time(start_time)} - {format_time(end_time)}] {segment['text']}\n"
        return transcript_with_timestamps, transcript_list
    except Exception as e:
        st.error("Transcript extraction failed. Please check the video link.")
        return None, None


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


# Function to predict timestamps for a given subtopic
def predict_timestamps(transcript_list, subtopic):
    # Extract text from transcript segments
    segments_text = [segment['text'] for segment in transcript_list]

    # Perform named entity recognition (NER) on the subtopic
    doc = nlp(subtopic)
    ner_subtopic = [ent.text for ent in doc.ents]

    # Keyword extraction for the subtopic
    keywords = [token.text for token in doc if not token.is_stop]

    # Combine NER entities and keywords
    subtopic_tokens = set(ner_subtopic + keywords)

    # Vectorize the segments and the subtopic tokens
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(segments_text)
    subtopic_vec = vectorizer.transform([" ".join(subtopic_tokens)])

    # Compute cosine similarity between subtopic and each segment
    similarities = cosine_similarity(X, subtopic_vec)

    # Get index of the segment with highest similarity
    closest_segment_index = np.argmax(similarities)

    # Get the start time of the closest segment
    closest_segment = transcript_list[closest_segment_index]
    start_time = closest_segment['start']
    return start_time


def generate_youtube_link(video_id, start_time_seconds):
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    time_parameter = f"&t={start_time_seconds}s"
    return base_url + time_parameter


def main():
    st.title("YouTube Link Generator")

    # Input
    video_link = st.text_input("Enter the YouTube video link:")
    keyword = st.text_input("Enter the keyword:")

    if video_link and keyword:
        video_id = video_link.split("v=")[-1]  # Extract video ID from link
        transcript_with_timestamps, transcript_list = extract_transcript_with_timestamps(video_id)

        if transcript_with_timestamps and transcript_list:
            # Generate YouTube link
            start_time = predict_timestamps(transcript_list, keyword)
            start_time_seconds = int(start_time)
            youtube_link = generate_youtube_link(video_id, start_time_seconds)
            st.success("YouTube link with timestamp:")
            st.write(youtube_link)
        else:
            st.error("Transcript extraction failed. Please try another video link.")

if __name__ == "__main__":
    main()
