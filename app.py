import streamlit as st
import json
import torch
import faiss
import requests
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from transformers import GPT2Tokenizer, GPT2Model
from graphviz import Digraph

from anthropic import Anthropic
from anthropic.types.message import Message

ANTHROPIC_API_KEY = ""
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_text(prompt):
  response: Message = client.messages.create(
      max_tokens=1000,
      messages=[{'role': 'user', 'content': prompt}],
      model="claude-3-opus-20240229",
      temperature=0.5,
  )

  answer = response.content[0].text
  return answer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def embed_code(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract hidden states from the output
    last_hidden_states = outputs.last_hidden_state
    # Use the mean of the hidden states as the embedding
    return torch.mean(last_hidden_states, dim=1).squeeze().numpy()

def generate_summary(macro_code):
    # Generate a prompt for summarizing the macro steps
    prompt = f"""Summarize the functionality of the following VBA macro code into 4-5 concise steps:

{macro_code}"""

    # Generate summary using Claude API (replace with your text generation logic)
    summary = generate_text(prompt)

    return summary

def generate_description(macro_code, index, metadata):
    # Find the closest embeddings for code part only
    macro_code_embedding = embed_code(macro_code)
    distances, indices = index.search(np.array([macro_code_embedding]), k=5)
    context_examples = [metadata[i] for i in indices[0]]

    # Create context string
    context = "\n\n".join([f"Title: {ex['title']}\nDescription: {ex['description']}\nCode: {ex['code']}" for ex in context_examples])

    # Format combined prompt
    combined_prompt = f"""You are a knowledgeable assistant that can provide detailed descriptions of VBA macros.
You have access to a database of macros with titles, descriptions, and code snippets.

{context}

Given the following macro code, provide a detailed description of its functionality:

{macro_code}"""
    detailed_description = generate_text(combined_prompt)

    return detailed_description


def load_data_and_create_index():
    # Load the JSON file
    with open('macros.json', 'r') as file:
        data = json.load(file).get('macros')

    # Extract Title, Description, and Code fields
    examples = []
    for item in data:
        title = item.get('title')
        description = item.get('description')
        code = "\n".join(item.get('code'))  # Join the code list into a single string
        if code!='':
            examples.append({
                'title': title,
                'description': description,
                'code': code
            })
    
    # Create embeddings for each example (only for code part)
    for example in examples:
        example['code_embedding'] = embed_code(example['code'])

    # Combine embeddings (only code part)
    embeddings = np.array([example['code_embedding'] for example in examples],dtype='float32')

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Store example metadata
    metadata = [{'title': example['title'], 'description': example['description'], 'code': example['code']} for example in examples]
    
    return index, metadata

def generate_process_flow(description, macro_name):
    dot = Digraph(comment=macro_name)
    lines = description.strip().split('. ')
    prev_node = None

    for i, line in enumerate(lines):
        node_name = f'node{i}'
        dot.node(node_name, line.strip())
        if prev_node:
            dot.edge(prev_node, node_name)
        prev_node = node_name

    return dot

def display_process_flow(dot, macro_name, output_dir='output'):
    st.graphviz_chart(dot)


def analyze_vba_macro(code):
    # Remove comments (assuming single-line comments starting with ')
    code = re.sub(r"'.*", "", code)

    # Count lines of code
    lines_of_code = len(code.strip().splitlines())

    # Count variables declared
    variables_declared = re.findall(r"Dim\s+(\w+)\s+As", code)

    # Count loops (For loops)
    loops_count = len(re.findall(r"\bFor\b", code))

    # Detect inefficient loops (e.g., nested loops or loops iterating over large ranges)
    inefficient_loops = re.findall(r"\bFor\s+\w+\s+=\s+1\s+To\s+\d+\b", code)

    # Calculate cyclomatic complexity (simplified)
    complexity = loops_count + 1  # Basic calculation: loops + 1

    # Basic recommendations
    recommendations = []
    if complexity > 5:
        recommendations.append("Consider refactoring to reduce complexity.")
    if len(inefficient_loops) > 0:
        recommendations.append("Review inefficient loops.")

    return {
        "Lines of Code": lines_of_code,
        "Variables Declared": len(variables_declared),
        "Loops Count": loops_count,
        "Cyclomatic Complexity": complexity,
        "Recommendations": recommendations
    }

# Load data and create FAISS index
index, metadata = load_data_and_create_index()

# Streamlit UI
st.title("VBA Macro Description Generator")
st.write("Enter VBA Macro Code to get a detailed description:")

macro_code_input = st.text_area("VBA Macro Code", height=200)

if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = None

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Generate Description"):
        st.session_state.button_clicked = "Button 1"
with col2:
    if st.button("Visualize logic flow"):
        st.session_state.button_clicked = "Button 2"
with col3:
    if st.button("Analyze Macro"):
        st.session_state.button_clicked = "Button 3"

if 'button_clicked' in st.session_state:
    if st.session_state.button_clicked == "Button 1":
        if macro_code_input.strip() != "":
            st.write("## Generated Description")
            description = generate_description(macro_code_input, index, metadata)
            st.markdown(description)
        else:
            st.error("Please enter some VBA macro code.")

    elif st.session_state.button_clicked == "Button 2":
        summary = generate_summary(macro_code_input)
        dot = generate_process_flow(summary, "Description")
        display_process_flow(dot, "Flow Chart")
        

    elif st.session_state.button_clicked == "Button 3":
        analysis_result = analyze_vba_macro(macro_code_input)
        st.write("VBA Macro Analysis Results:")
        for key, value in analysis_result.items():
            if key == "Recommendations":
                if len(value) > 0:
                    st.write(f"- {key}:")
                    for rec in value:
                        st.write(f"  - {rec}")
                else:
                    st.write(f"- {key}: None")
            else:
                st.write(f"- {key}: {value}")   