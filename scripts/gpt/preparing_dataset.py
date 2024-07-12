import pandas as pd
from datasets import Dataset
import json

# Function to prepare each example
def prepare_examples(example):
    return {
        "sent_id": example["sent_id"],
        "content" :{
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": example["sentence"]}
        ]}
    }

domain = "general-domain"

# Read CSV files
test_df = pd.read_csv(f"datasets/{domain}/test_set.txt")
test_df.head()

# Convert DataFrame to Dataset
test_dataset = Dataset.from_pandas(test_df)

# Apply the prepare_examples function
test_dataset = test_dataset.map(prepare_examples, remove_columns=['sentence', 'entities', 'uris'])

# Extract only the messages field
messages_list = [example for example in test_dataset]

# Save the messages to a JSONL file
json_output = f"data/{domain}/test_final.jsonl"
with open(json_output, "w") as json_file:
    for message in messages_list:
        json.dump(message, json_file)
        json_file.write("\n")

print(f"Messages saved to {json_output}")
