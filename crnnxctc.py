import sys
sys.path.append('.')

import pandas as pd

import tensorflow as tf
import random
random.seed(2022)

from evaluator import Evaluator
from loader import DataImporter
from loader import DataHandler
from visualizer import visualize_images_labels, plot_training_results
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, GRU
from layers import custom_cnn, reshape_features
from models import get_imagenet_model
from tensorflow.keras.callbacks import EarlyStopping
from losses import CTCLoss
from metrics import SequenceAccuracy, CharacterAccuracy, LevenshteinDistance
from tensorflow.keras.optimizers import Adadelta

tf.get_logger().setLevel('ERROR')
APPROACH_NAME = 'CRNNxCTC'

"""# Data input pipeline"""

DATASET_DIR = r'./Patches'
ALL_TRANSCRIPTS_PATH = f'{DATASET_DIR}/All.txt'
VALID_TRANSCRIPTS_PATH = f'{DATASET_DIR}/Validate.txt'
FONT_PATH = r'./NomNaTong-Regular.ttf'

"""## Load and remove records with rare characters"""


dataset = DataImporter(DATASET_DIR, ALL_TRANSCRIPTS_PATH, min_length=1)
print(dataset)

"""## Data constants and input pipeline"""

HEIGHT, WIDTH = 432, 48
PADDING_CHAR = '[PAD]'
BATCH_SIZE = 32

data_handler = DataHandler(dataset, img_size=(HEIGHT, WIDTH), padding_char=PADDING_CHAR)
NUM_VALIDATE = DataImporter(DATASET_DIR, VALID_TRANSCRIPTS_PATH, min_length=1).size
VOCAB_SIZE = data_handler.char2num.vocab_size()

"""## Visualize the data"""

visualize_images_labels(
    dataset.img_paths,
    dataset.labels,
    figsize = (15, 15),
    subplot_size = (2, 8),
    font_path = FONT_PATH
)

"""# Define the model"""

def build_crnn(imagenet_model=None, imagenet_output_layer=None, name='CRNN'):
    if imagenet_model:
        image_input = imagenet_model.input
        imagenet_model.layers[0]._name = 'image'
        x = imagenet_model.get_layer(imagenet_output_layer).output
    else:
        image_input = Input(shape=(HEIGHT, WIDTH, 3), dtype='float32', name='image')
        conv_blocks_config = {
            'block1': {'num_conv': 1, 'filters':  64, 'pool_size': (2, 2)},
            'block2': {'num_conv': 1, 'filters': 128, 'pool_size': (2, 2)},
            'block3': {'num_conv': 2, 'filters': 256, 'pool_size': (2, 2)},
            'block4': {'num_conv': 2, 'filters': 512, 'pool_size': (2, 2)},

            'block5': {'num_conv': 2, 'filters': 512, 'pool_size': None},
        }
        x = custom_cnn(conv_blocks_config, image_input)

    feature_maps = reshape_features(x, dim_to_keep=1, name='rnn_input')

    bigru1 = Bidirectional(GRU(256, return_sequences=True), name='bigru1')(feature_maps)
    bigru2 = Bidirectional(GRU(256, return_sequences=True), name='bigru2')(bigru1)

    y_pred = Dense(
        units = VOCAB_SIZE + 1, 
        activation = 'softmax',
        name = 'rnn_output'
    )(bigru2)
    return Model(inputs=image_input, outputs=y_pred, name=name)

imagenet_model, imagenet_output_layer = None, None

model = build_crnn(imagenet_model, imagenet_output_layer)
model.summary(line_length=110)

"""# Training"""

train_idxs = list(range(dataset.size - NUM_VALIDATE))
valid_idxs = list(range(train_idxs[-1] + 1, dataset.size))
print('Number of training samples:', len(train_idxs))
print('Number of validate samples:', len(valid_idxs))
random.shuffle(train_idxs)
train_tf_dataset = data_handler.prepare_tf_dataset(train_idxs, BATCH_SIZE)
valid_tf_dataset = data_handler.prepare_tf_dataset(valid_idxs, BATCH_SIZE)

"""## Callbacks"""

early_stopping_callback = EarlyStopping(
    monitor = 'val_loss',
    min_delta = 1e-3,
    patience = 5,
    restore_best_weights = True,
    verbose = 1
)

"""## Train the NomNaOCR dataset"""

LEARNING_RATE = 1.0
EPOCHS = 100

model.compile(
    optimizer = Adadelta(LEARNING_RATE),
    loss = CTCLoss(),
    metrics = [
        SequenceAccuracy(use_ctc_decode=True),
        CharacterAccuracy(use_ctc_decode=True),
        LevenshteinDistance(use_ctc_decode=True, normalize=True)
    ]
)

history = model.fit(
    train_tf_dataset,
    validation_data = valid_tf_dataset,
    epochs = EPOCHS,
    callbacks = [early_stopping_callback],
    verbose = 1
).history

"""## Save the training results"""

best_epoch = early_stopping_callback.best_epoch
print(f'- Loss on validation\t: {history["val_loss"][best_epoch]}')
print(f'- Sequence accuracy\t: {history["val_seq_acc"][best_epoch]}')
print(f'- Character accuracy\t: {history["val_char_acc"][best_epoch]}')
print(f'- Levenshtein distance\t: {history["val_levenshtein_distance"][best_epoch]}')

plot_training_results(history, f'{APPROACH_NAME}.png')
model.save_weights(f'{APPROACH_NAME}.h5')

"""# Inference"""

reset_model = build_crnn(imagenet_model, imagenet_output_layer)
reset_model.load_weights(f'CRNNxCTC.h5')
reset_model.summary(line_length=110)
LEARNING_RATE = 1.0
reset_model.compile(
    optimizer = Adadelta(LEARNING_RATE),
    loss = CTCLoss(),
    metrics = [
        SequenceAccuracy(use_ctc_decode=True),
        CharacterAccuracy(use_ctc_decode=True),
        LevenshteinDistance(use_ctc_decode=True, normalize=True)
    ]
)

"""## On test dataset"""

for idx, (batch_images, batch_tokens) in enumerate(valid_tf_dataset.take(1)):
    idxs_in_batch = valid_idxs[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
    labels = data_handler.tokens2texts(batch_tokens)
    pred_tokens = reset_model.predict(batch_images)
    pred_labels = data_handler.tokens2texts(pred_tokens, use_ctc_decode=True)

    visualize_images_labels(
        img_paths = dataset.img_paths[idxs_in_batch],
        labels = labels,
        pred_labels = pred_labels,
        figsize = (11.6, 30),
        subplot_size = (4, 8),
        legend_loc = (3.8, 4.38),
        annotate_loc = (4, 2.75),
        font_path = FONT_PATH,
    )
    print(
        f'Batch {idx + 1:02d}:\n'
        f'- True: {dict(enumerate(labels, start=1))}\n'
        f'- Pred: {dict(enumerate(pred_labels, start=1))}\n'
    )

"""## On random image"""

random_path = './囷𦝄苔惮󰞺𧍋𦬑囊.jpg'
random_label = '囷𦝄苔惮󰞺𧍋𦬑囊'
random_image = data_handler.process_image(random_path)
pred_tokens = reset_model.predict(tf.expand_dims(random_image, axis=0))
pred_labels = data_handler.tokens2texts(pred_tokens, use_ctc_decode=True)

visualize_images_labels(
    img_paths = [random_path],
    labels = [random_label],
    pred_labels = pred_labels,
    figsize = (5, 4),
    subplot_size = (1, 1),
    font_path = FONT_PATH,
)
print('Predicted text:', ''.join(pred_labels))

"""# Detail evaluation"""

GT10_TRANSCRIPTS_PATH = f'{DATASET_DIR}/Validate_gt10.txt'
LTE10_TRANSCRIPTS_PATH = f'{DATASET_DIR}/Validate_lte10.txt'

gt10_evaluator = Evaluator(reset_model, DATASET_DIR, GT10_TRANSCRIPTS_PATH)
lte10_evaluator = Evaluator(reset_model, DATASET_DIR, LTE10_TRANSCRIPTS_PATH)

df = pd.DataFrame([
    reset_model.evaluate(valid_tf_dataset, return_dict=True),
    gt10_evaluator.evaluate(data_handler, BATCH_SIZE),
    lte10_evaluator.evaluate(data_handler, BATCH_SIZE),
])

df.index = ['Full', 'Length > 10', 'Length ≤ 10']

print(df)