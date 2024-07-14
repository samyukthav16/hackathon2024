# hackathon2024

# VBA Macro Analyzer App

## Overview

The VBA Macro Analyzer App is a Streamlit-based application designed to help users understand and analyze VBA macro code. The app provides detailed descriptions, visualizations, and analyses of VBA macros to improve code comprehension and quality.

## Features

1. *Generate Description*: 
   - Generates a detailed description of the provided VBA macro code using the Claude API for natural language processing.
   
2. *Visualize Logic Flow*: 
   - Creates a visual representation of the macro's logic flow using Graphviz to help users understand the code structure and execution flow.
   
3. *Analyze Macro*:
   - Analyzes the VBA macro for various metrics, such as lines of code, variables declared, loops count, cyclomatic complexity, and provides recommendations for improvement.

## Installation

To run the VBA Macro Analyzer App locally, follow these steps:

1. Clone the repository:
   bash
   git clone https://github.com/yourusername/vba-macro-analyzer.git
   cd vba-macro-analyzer
   

2. Install the required dependencies:
   bash
   pip install -r requirements.txt
   

3. Run the Streamlit app:
   bash
   streamlit run app.py
   

## Usage

1. *Enter VBA Macro Code*:
   - Paste your VBA macro code into the text area provided in the app.

2. *Generate Description*:
   - Click the "Generate Description" button to receive a detailed explanation of the macro's functionality.

3. *Visualize Logic Flow*:
   - Click the "Visualize logic flow" button to generate a flowchart of the macro's logic.

4. *Analyze Macro*:
   - Click the "Analyze Macro" button to get an analysis report of the macro, including metrics and recommendations.

## Dependencies

- Streamlit
- transformers (GPT-2)
- torch (PyTorch)
- faiss
- numpy
- graphviz
- re
- requests
- matplotlib
