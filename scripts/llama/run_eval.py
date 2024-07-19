import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

# Set display options
pd.set_option('display.max_colwidth', None)  # or use a large value like 1000

# Process the 'linking' column to extract entities and uris
def process_linking(linking_str):
    linking_str = linking_str.replace("Here are the entities and their corresponding entry links in Wikidata:,", "")
    linking_str = linking_str.replace("[", "")
    linking_str = linking_str.replace("]", "")
    linking_str = linking_str.strip()
    entities, uris = [], []
    if linking_str != "" :
        for pair in linking_str.split(", "):
            if len(pair.split("=")) > 2:
                entity = pair.split("=")[-2]
                uri = pair.split("=")[-1]
            else:
                entity, uri = pair.split("=")
                uri = uri.replace("Wikidata:", "")
                uri = uri.replace("simple:", "")
                uri = uri.replace("https://www.wikidata.org/wiki/", "")
                uri = uri.strip()
                if len(uri.split(" ")) > 1:
                    uri = uri.split(" ")[0]
                uri = uri.strip()
            entities.append(entity.strip())
            uris.append(uri)
    if len(entities)==0:
        entities.append("empty")
        uris.append("empty")
    return entities, uris

# Function to extract the code from the URI
def extract_code(uri):
    return uri.split('/')[-1]

def compute_metrics_per_pair(true_pairs, pred_pairs):
    precision_list = []
    recall_list = [] 
    f1_list = []
    total_pairs = 0  # To calculate accuracy
    for true_pair_list, pred_pair_list in zip(true_pairs, pred_pairs):
        true_counter = Counter(true_pair_list)
        pred_counter = Counter(pred_pair_list)
        
        true_positive = sum((true_counter & pred_counter).values())  # Intersection: min of counts for each entity
        false_positive = sum((pred_counter - true_counter).values())  # Predicted but not in true
        false_negative = sum((true_counter - pred_counter).values())  # In true but not predicted
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    precision_avg = np.mean(precision_list)
    recall_avg = np.mean(recall_list)
    f1_avg=np.mean(f1_list)
    return precision_avg, recall_avg, f1_avg

# Function to find correct entity-URI pairs in linking_df that match the test_df (ground truth)
def get_correct_entity_uri_pairs(linking_df, test_df):
    correct_pairs_per_sent_id = []
    
    for index, row in linking_df.iterrows():
        sent_id = row['sent_id']
        linking_entities = row['linking_entities']
        linking_uris = row['linking_uris']
        
        # Get corresponding entities and URIs from test_df
        test_row = test_df[test_df['sent_id'] == sent_id]
        if not test_row.empty:
            test_entities = test_row.iloc[0]['entities']
            test_uris = test_row.iloc[0]['uris']
            
            # Create pairs for comparison
            linking_pairs = set(zip(linking_entities, linking_uris))
            test_pairs = set(zip(test_entities, test_uris))
            
            # Find correct pairs
            correct_pairs = linking_pairs & test_pairs
            A=False
            if len(correct_pairs)>0:
                if len(correct_pairs)==len(test_pairs):
                   A=True 
            correct_pairs_per_sent_id.append({
                'sent_id': sent_id,
                'correct_entity_uri_pairs': list(correct_pairs),
                'count_correct_entity_uri_pairs': len(correct_pairs),
                'prediction_truth uri pairs': list(linking_pairs),
                'ground_truth_uri_pairs': list(test_pairs),
                'count_ground_truth_uri_pairs': len(test_pairs),
                'exact_match': A,
            })
    
    return correct_pairs_per_sent_id

domain = "general-domain"
learning_method = "fine-tuning"
model = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = model.split("/")[-1]

print("Evaluation EL prediction results started ...")
print("Initial information")
print(f"Domain: {domain}")
print(f"Learning Method: {learning_method}")
print(f"Model Name: {model_name}")

print("Loading EL prediction results")
# Read the prediction-results file into a pandas DataFrame
linking_df = pd.read_csv(f"../../results/performance-analysis/{domain}/{learning_method}/{model_name}/prediction-results.txt", sep="\t", header=None, names=["sent_id", "linking"])
print(linking_df.head())

print("Loading IndEL dataset")
# Read the test file into a pandas DataFrame
test_df = pd.read_csv(f"../../datasets/{domain}/test_set.txt")
print(test_df.head())

# Convert strings to lists for comparison
test_df['entities'] = test_df['entities'].apply(eval)
test_df['uris'] = test_df['uris'].apply(eval)
# Apply the function to the uris column and create a new column 'uri_codes'
#test_df['linking_uris'] = test_df['uris'].apply(lambda uri_list: [extract_code(uri) for uri in uri_list])

print("Comparing process ...")
linking_df[['linking_entities', 'linking_uris']] = linking_df['linking'].apply(lambda x: pd.Series(process_linking(x)))
linking_df.drop(columns=['linking'], inplace=True)

# Get the correct entity-URI pairs per sent_id
correct_entity_uri_pairs_list = get_correct_entity_uri_pairs(linking_df, test_df)

# Convert the list of dictionaries to a DataFrame
correct_entity_uri_pairs_df = pd.DataFrame(correct_entity_uri_pairs_list)

# Adding a column that contains pairs of entities and uris
linking_df['entity_uri_pairs'] = linking_df.apply(lambda row: list(zip(row['linking_entities'], row['linking_uris'])), axis=1)
test_df['entity_uri_pairs'] = test_df.apply(lambda row: list(zip(row['entities'], row['uris'])), axis=1)

# Compute the metrics for entity-URI pairs
precision_pairs, recall_pairs, f1_pairs= compute_metrics_per_pair(test_df['entity_uri_pairs'], linking_df['entity_uri_pairs'])

print("EL prediction results")
print(f"Precision: {precision_pairs}")
print(f"recall: {recall_pairs}")
print(f"f1-score: {f1_pairs}")
