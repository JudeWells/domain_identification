import os

from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# The examples in this notebook use a set of nine benchmarks described in our publication.
# These benchmarks can be downloaded via FTP from: ftp.cs.huji.ac.il/users/nadavb/protein_bert/protein_benchmarks
# Download the benchmarks into a directory on your machine and set the following variable to the path of that directory.
BENCHMARKS_DIR = 'datasets/PPI/'
BENCHMARK_NAME = 'protbert_ppi'
MODEL_SAVE_DIR = 'proteinbert_models/'

# A local (non-global) bianry output
OUTPUT_TYPE = OutputType(True, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)


# Loading the dataset
train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.train.csv' % BENCHMARK_NAME)
train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
train_set, valid_set = train_test_split(train_set, test_size = 0.1, random_state = 0)
# valid_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.valid.csv' % BENCHMARK_NAME)
# valid_set = pd.read_csv(valid_set_file_path).dropna().drop_duplicates()

test_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.test.csv' % BENCHMARK_NAME)
test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')


# Loading the pre-trained model and fine-tuning it on the loaded dataset

pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir=MODEL_SAVE_DIR,
                                                                    local_model_dump_file_name='epoch_92400_sample_23500000.pkl',
                                                                  )

# get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
        get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

model_filepath = f"{MODEL_SAVE_DIR}/ppi_model_{0}"

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience = 2, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    # keras.callbacks.ModelCheckpoint(model_filepath, monitor="val_loss", save_best_only=True)
]

# max epochs per stage = 40
# final seq_len = 1024

finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
        seq_len = 512, batch_size = 32, max_epochs_per_stage = 60, lr = 1e-04, begin_with_frozen_pretrained_layers = True, \
        lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 1024, final_lr = 1e-05, callbacks = training_callbacks)


# Evaluating the performance on the test-set

results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], \
        start_seq_len = 512, start_batch_size = 32)
results.to_csv('ppi_results.csv', index=True)
bp = True