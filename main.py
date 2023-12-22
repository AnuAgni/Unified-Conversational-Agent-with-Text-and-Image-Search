from dotenv import load_dotenv
import os
from llama_index import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import streamlit as st

# Load the environment variables
load_dotenv()

# Secured Key
openai_key = os.environ['OPENAI_API_TOKEN']

# LLM Model
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview",
    api_key=openai_key,
    max_new_tokens=500,
    temperature=0.0,
)

# Function to extract text from the input image
def text_from_image(image_path):
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()
    query = """
        Extract all text from the provided image and provide the same as output in a structured format.
    """
    response = openai_mm_llm.complete(
        prompt=query,
        image_documents=image_document
    )
    return response

# Function to extract details and information about a chart from the input image
def chart_from_image(image_path):
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()
    query = """
        Analyze the graph in detail, breakdown all the elements and also provide the accurate inferences or valuable insights obtained from the graph.
    """
    response = openai_mm_llm.complete(
        prompt=query,
        image_documents=image_document
    )
    return response

# Function to extract table and all relevant details and inferences from the input image
def table_from_image(image_path):
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()
    query = """
        Extract all text from the provided image and display the result accurately in structured format as a table.
    """
    response = openai_mm_llm.complete(
        prompt=query,
        image_documents=image_document
    )
    return response

# Function to extract formulae and all relevant details from the input image
def formulae_from_image(image_path):
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()
    query = """
        Analyze the image and extract the formula with all relevant details for explaining the same.
    """
    response = openai_mm_llm.complete(
        prompt=query,
        image_documents=image_document
    )
    return response
    

if __name__ == '__main__':
    st.title('POC on Image Analysis Bot')
    selected_option = st.sidebar.selectbox("Select the type of image you want to query with:", ["Text", "Chart", "Table", "Formula"])
    
    # Defining all the images for testing
    text_image_1 = "images/text-image1.jpg"
    text_image_2 = "images/text-image2.png"
    chart_image = "images/chart.png"
    table_image = "images/table.png"
    formula_image_1 = "images/text_figure_formula.jpg"
    formula_image_2 = "images/figure_formula.jpeg"

    feedback_message = st.empty()
    # Based on the selection we can obtain the results
    if selected_option == "Text":
        st.image(text_image_1, use_column_width=True)
        feedback_message.text("Results are getting fetched...")
        response = text_from_image(text_image_1)
        feedback_message.text(response)
    
    if selected_option == "Chart":
        st.image(chart_image, use_column_width=True)
        feedback_message.text("Results are getting fetched...")
        response = chart_from_image(chart_image)
        feedback_message.text(response)

    if selected_option == "Table":
        st.image(table_image, use_column_width=True)
        feedback_message.text("Results are getting fetched...")
        response = table_from_image(table_image)
        feedback_message.text(response)

    if selected_option == "Formula":
        st.image(formula_image_1, use_column_width=True)
        feedback_message.text("Results are getting fetched...")
        response = formulae_from_image(formula_image_1)
        feedback_message.text(response)