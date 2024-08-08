from openai import OpenAI
import gradio as gr

# Set your OpenAI API key
client = OpenAI(api_key='sk-None-lWf579FvEQUJbg6EAjWYT3BlbkFJv1pnFpxdh2pJV52yXP4k')

def create_system_task():
    """
    Create the system-level task for the OpenAI model.

    Returns:
        str: The system task string.
    """
    return (
        "Summarize and analyze the most relevant information related to the stock data provided "
        "for that day in a conversational manner. Provide a detailed explanation including recommendation, "
        "investment horizon, and justification that would be understandable for the general public. "
        "Format the response as a detailed explanation.\n"
        "Also, give the approximate price prediction for the next couple days, next week, next month, next 3 months, next 6 months, next year, and next 5 years.\n"
        "Do not say something like you cannot predict price or it has lots of different factors; in respect to the knowledge that you have, give estimation for sure."
    )

def create_user_prompt(user_input, stock_data):
    """
    Create the user prompt for the OpenAI model.

    Args:
        user_input (str): The input from the user.
        stock_data (dict): The stock data.

    Returns:
        str: The formatted user prompt.
    """
    return f"{user_input}\n\nGiven the following stock data: {stock_data}\n Also try to el"

def predict(message, history):
    """
    Generate a response from the OpenAI model based on the user's message and conversation history.

    Args:
        message (str): The user's message.
        history (list): The conversation history.

    Yields:
        str: The partial response from the OpenAI model.
    """
    # Stock data placeholder, should be replaced with actual data
    stock_data = None  # "{data}"

    # Create a base prompt with system-level instructions
    base_prompt = [
        {"role": "system", "content": "You are a helpful assistant that helps users make informed decisions regarding the stock with respect to the information that they are going to provide."}
    ]

    # Create and add the system task
    system_task = create_system_task()
    base_prompt.append({"role": "system", "content": system_task})

    # Convert the history into the format required by OpenAI
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})

    # Create and add the user prompt
    if stock_data:
        user_prompt = create_user_prompt(message, stock_data)
    else:
        user_prompt = 'response to the following user message in details, never say you cannot predict or it is better to use other indicators for analysis. Always answer with confidence. User message:\n\n' + message
            
    history_openai_format.append({"role": "user", "content": user_prompt})

    # Combine the base prompt with the history
    full_prompt = base_prompt + history_openai_format
    print(full_prompt)

    # Generate a response from the OpenAI API
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::9tnxflRG",
        messages=full_prompt,
        temperature=0.3,  # Adjusted temperature for more controlled responses
        max_tokens=4050,  # Adjust as needed to control response length
        stream=True
    )

    # Yield the response in chunks for streaming
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

# Launch the Gradio interface
gr.ChatInterface(predict).launch()
