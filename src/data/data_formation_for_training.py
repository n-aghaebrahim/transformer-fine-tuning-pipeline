import pandas as pd
import json
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key='xxxxx')

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('AAPL_MasterData-2018-2023.csv')

# Extract the specific columns
close_next_1_day = df['Close_Next_1_Day'].tolist()
close_next_5_days = df['Close_Next_5_Days'].tolist()
close_next_30_days = df['Close_Next_30_Days'].tolist()
close_next1_pctchg = df['Close_Next1_pctchg'].tolist()
close_next5_pctchg = df['Close_Next5_pctchg'].tolist()
close_next30_pctchg = df['Close_Next30_pctchg'].tolist()

# Drop the specific columns from the DataFrame
df.drop(columns=[
    'Close_Next_1_Day', 'Close_Next_5_Days', 'Close_Next_30_Days',
    'Close_Next1_pctchg', 'Close_Next5_pctchg', 'Close_Next30_pctchg'
], inplace=True)

# Limit the DataFrame to the first 1200 rows
df = df[:1200]

def get_summary_and_analysis(data):
    """
    Function to get summary and analysis from OpenAI.

    Args:
        data (dict): The stock data for a specific date.

    Returns:
        dict: The response from OpenAI containing the summary and analysis.
    """
    prompt = (f"Given the following stock data: {data}, summarize and analyze the most relevant information "
              "related to the stock for that day in a conversational manner. Provide a detailed explanation including "
              "recommendation, investment horizon, and justification that would be understandable for the general public. "
              "Format the response as a detailed explanation rather than just key-value pairs.")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You help to summarize stock data for each date."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=500,
        top_p=1
    )
    return response.choices[0].message

# Create a list of dictionaries with the new structure
dataset = []
for i, record in df.iterrows():
    data_dict = record.to_dict()
    analysis = get_summary_and_analysis(data_dict)
    summary = analysis.content
    target_dict = {
        "summary": summary,
        "recommendation": None,
        "prediction": (f"The possibility of price change for next day is {close_next_1_day[i]}, "
                       f"for next 5 days is {close_next_5_days[i]}, "
                       f"for next 30 days is {close_next_30_days[i]}, with percentage changes of "
                       f"{close_next1_pctchg[i]}, {close_next5_pctchg[i]}, and {close_next30_pctchg[i]}, respectively.")
    }
    dataset.append({"data": data_dict, "target": target_dict})

# Display the first few records to verify
for record in dataset[:5]:
    print(record)

def fill_recommendation(data):
    """
    Function to get a recommendation from OpenAI based on the stock data and target.

    Args:
        data (dict): The stock data and target for a specific date.

    Returns:
        dict: The response from OpenAI containing the recommendation.
    """
    prompt = (f"Given the following stock data: {data}, analyze the most relevant information in the target dict "
              "and data dict, then try to fill out the recommendation. Make sure the recommendation is short and "
              "mentions whether to buy or sell in the short term or long term.")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You help to summarize stock data for each date."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=500,
        top_p=1
    )
    return response.choices[0].message

# Update the dataset with recommendations
for record in dataset:
    analysis = fill_recommendation(record)
    record['target']['recommendation'] = analysis.content
    print(record)

# Save the dataset to a JSON file
with open('stock_data_analysis_train.json', 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

# Load the JSON file
with open('stock_data_analysis_train.json', 'r') as f:
    data = json.load(f)

# Open a new file to write in .jsonl format
with open('apple_stock_train_data.jsonl', 'w') as f:
    for item in data:
        # Extract the data and target fields
        data_content = json.dumps(item['data'])  # Convert data to JSON string
        target_content = item['target']  # Assuming target is already a string

        # Create the structured message
        message = {
            "messages": [
                {"role": "system", "content": "You are a helpful financial analyst, you help users to make the right investment decisions based on the data and information provided."},
                {"role": "user", "content": f"Is it a good time to invest in stock? Here are the financial information and news:\n\n{data_content}."},
                {"role": "assistant", "content": f"Based on today's information, here is the analysis:\n{target_content}"}
            ]
        }

        # Write each structured message as a JSON line
        f.write(json.dumps(message) + '\n')
