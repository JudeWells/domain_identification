import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

"""
This module finetunes the protein-bert model for domain residue classificaion and evaluates it on an unseen test set
The script is desiged to be called multiple times each with different training hyperparamaters which
are obtained from hyperparameters.csv an integer in range 1-len(hyperparameters.csv) should be passed
AUC ROC score on test set is saved for each model (both in the hyperparams.csv file and in a unique csv file for each
time the script is called. 
"""


DATA_DIR = 'data/'
FILENAME = 'protbert_domain'
MODEL_SAVE_DIR = '../protbert/proteinbert_models/'
PROTBERT_TRAINED_MODEL_FILENAME = 'epoch_92400_sample_23500000.pkl'

# A local (non-global) bianry output
OUTPUT_TYPE = OutputType(True, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
OUTPUT_DIR = 'output'


# Loading the dataset
train_set_file_path = os.path.join(DATA_DIR, '%s.train.csv' % FILENAME)
train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
train_set, test_set = train_test_split(train_set, test_size=0.25, random_state=0)
train_set, valid_set = train_test_split(train_set, test_size=0.25, random_state=0)


print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')
os.makedirs(OUTPUT_DIR, exist_ok=True)
completed = [int(f.split('_')[-1][:-4]) for f in os.listdir(OUTPUT_DIR)]
hyperparams = pd.read_csv('hyperparams.csv')


args = sys.argv
index = int(args[1]) -1


hp = hyperparams.loc[index].to_dict()
print(f'INDEX: {index}')
for k,v in hp.items():
    print(f'{k}: {v}')
# Loading the pre-trained model and fine-tuning it on the loaded dataset

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir=MODEL_SAVE_DIR,
                                                            local_model_dump_file_name=PROTBERT_TRAINED_MODEL_FILENAME,
                                                                  )
# get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
        get_model_with_hidden_layers_as_outputs, dropout_rate=hp['dropout'])

model_filepath = f"{MODEL_SAVE_DIR}/ppi_model_{0}"

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=hp['reduce_lr_patience'], factor=0.25, min_lr=1e-07, verbose=1),
    keras.callbacks.EarlyStopping(patience=hp['es_patience'], restore_best_weights=True),
    # keras.callbacks.ModelCheckpoint(model_filepath, monitor="val_loss", save_best_only=True)
]

finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
        seq_len=hp['seq_len'], batch_size=32, max_epochs_per_stage=hp['max_epochs_per_stage'], lr=hp['lr'], begin_with_frozen_pretrained_layers=hp['begin_with_frozen_pretrained_layers'], \
        lr_with_frozen_pretrained_layers=hp['lr_with_frozen_pretrained_layers'], n_final_epochs=hp['n_final_epochs'], final_seq_len=2048, final_lr=hp['final_lr'], callbacks=training_callbacks)

# Evaluating the performance on the test-set
results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], \
        start_seq_len=512, start_batch_size=32)
result = results.loc['All'].to_dict()
hp.update(result)
results_filename = f'{OUTPUT_DIR}/ppi_{index}.csv'
pd.DataFrame([hp]).to_csv(results_filename, index=False)
hp_df = pd.read_csv('hyperparams.csv')
roc_auc = result['AUC']
hp_df.loc[index, 'roc_auc'] = roc_auc
hp_df.to_csv('hyperparams.csv', index=False)