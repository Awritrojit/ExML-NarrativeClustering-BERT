import torch 
from transformers import BertTokenizer, BertModel
import numpy as np

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to tokenize and get activations for a given text
def get_bert_activations(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**tokens)
    activations = outputs.last_hidden_state
    
    # Get the cls tokens -> the first token in bert activations
    cls_activations = activations[:, 0, :]

    # Convert to a NumPy array for further analysis if needed
    cls_activations_np = cls_activations.detach().numpy()#.T
    return cls_activations_np

def get_bert_activations_all_layers(texts):
    # Tokenize the input texts
    tokens = tokenizer(texts, return_tensors='pt', truncation=True, padding=True)

    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)

    # Extract the last hidden states from all layers
    all_hidden_states = outputs.hidden_states # tuple of all hidden states; 
                                              # the first element is the last_hidden_state
        

    # Extract the CLS tokens from all layers
    cls_tokens = np.array([layer[:, 0, :].detach().numpy() for layer in all_hidden_states])
    #print(outputs.hidden_states[0].shape)

    return cls_tokens


# Function to process the dataframe and get activations for each summary
def process_dataframe(df):

    activations_dict = {}

    cols = list(df.columns[1:]) # ['chatgpt', 'shakespeare', 'dickens', 'cbronte', 'conandoyle',
                                #   'poe', 'martin', 'saki', 'ohenry', 'brown', 'jkj']
    narrative_names = df['NarrativeName'].unique()

    for narrative in narrative_names:

        variants_df = df[df['NarrativeName'] == narrative]
        activations_dict_variants = {}

        for i, row in variants_df.iterrows():
            summaries = row[cols]
            
            activations_dict_variants[i%10] = {}

            for style, summary in zip(cols, summaries):
                activations = get_bert_activations_all_layers(summary)
                #activations = get_bert_activations(summary)
                activations_dict_variants[i%10][style] = activations
        
        activations_dict[narrative] = activations_dict_variants

    return activations_dict

