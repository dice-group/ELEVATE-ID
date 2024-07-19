# ELMEval: Evaluating Large Language Models in Entity Linking for Low-Resource Languages: Insights from Indonesian

## Short description
An evaluation framework of LLMs, including GPT-4 and Llama-3, in the EL task for LrLs, with a focus on Indonesian.

## Usage

### Datasets

Datasets comprise two different kinds of datasets derived from [IndEL dataset](https://github.com/dice-group/IndEL):
1. General-domain: consists of train, validation and test set.
2. Specific-domain: consists of train, validation and test set.

**Note:**: Rasio for spliting the datasets are a train set of 80%, and for 10% for each validation and test set. 

### Instruction Template

| **Instruction Template** |                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------|
| **Task Description**     | Find entities and their corresponding entry links in Wikidata within the following sentence. Use the context of the sentence to determine the correct entries in Wikidata. |
| **Output Format**        | The output should be formatted as: [entity1=link1, entity2=link2]. No explanations are needed.|
| **Sample Sentence**      | Pria kelahiran Bogor, 16 Maret 60 tahun silam itu juga ditunjuk sebagai salah satu direktur Indofood dalam RUPS Juni 2008 silam. (A man born in Bogor, 60 years ago on March 16, was also appointed as one of the directors of Indofood in the General Meeting of Shareholders in June 2008.) |

### LLMs with Zero-shot learning
In the zero-shot setting, we prompt the LLMs using an instruction format shown in Table of Instruction Template, where the prompt includes only the task description and output format.

#### Generative Pre-training Transformer (GPT)
1. **Model:** GPT-4
2. **Preparing datasets**
Please, consider to update the domain manually on ```preparing_dataset.[y```
```
domain = "general-domain" # change to 'specific-domain' if you want to generata test dataset for specifice domain and vice-versa
```
```
cd scripts/gpt
python preparing_dataset.py
```
3. **Execute zero-shot learning prediction**
```
cd scripts/gpt
python run_predictions.py
```

#### LLMs of Llama Family
1. **Model:** Llama-3-8B-Instruct, Merak-7B-v4
2. **Preparing datasets**
   The datasets are available in datasets directory 
4. **Execute zero-shot learning prediction**
Please consider the value of ```domain``` and ```base_model_name``` subject to change.
```
cd scripts/llama
python run_predictions.py
```

### LLMs with Fine-tuning
In the fine-tuning setting, we provided the LLMs with prompts based on the fully detailed instructions shown in Table of Instruction Templat, where the example sentences are taken from the dataset.

1. **Preparing dataset:** To prepare the dataset for training the GPT model, refer to step 2 in the GPT section of "LLMs with Zero-shot Learning." Please consider the path of the source dataset, the domain, and the name of the file to store the processed data.
2. **Fine-tuning process**
- **GPT-3.5** We use GPT-3.5 to fine-tune the model due to its availability. To perform this process, you can follow the procedure at [OpenAI's fine-tuning platform](https://platform.openai.com/finetune). In this experiment, we set the hyperparameters as follows: number of epochs = 3, batch size = 8, and learning rate multiplier = 2.
- **Llama Family**
Please consider the value of ```domain``` and ```base_model_name``` subject to change.
```
cd scripts/llama
python llm-finetuning.py
```