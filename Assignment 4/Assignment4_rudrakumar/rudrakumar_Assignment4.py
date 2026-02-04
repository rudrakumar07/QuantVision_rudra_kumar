import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, to_categorical, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# CONFIG 
SIZE = 64
EPOCHS = 100
BATCH = 32

stocks = ['AAPL','GOOGL','MSFT','AMZN','TSLA','NVDA','AMD','INTC','NFLX','META']
train_folder = "images_train"
test_folder = "images_test"

np.random.seed(42)
tf.random.set_seed(42)

# FOLDERS 
for f in [train_folder, test_folder]:
    for lbl in ['Hammer','Doji','None']:
        os.makedirs(os.path.join(f, lbl), exist_ok=True)

# PATTERN LOGIC 
def check_pattern(row):
    body = abs(row['Open'] - row['Close'])
    lower = min(row['Open'], row['Close']) - row['Low']
    upper = row['High'] - max(row['Open'], row['Close'])
    total = row['High'] - row['Low']
    if total == 0:
        return 'None'
    if lower >= 2 * body and upper <= 0.5 * body:
        return 'Hammer'
    if body <= 0.1 * total:
        return 'Doji'
    return 'None'

# IMAGE DRAW 
def draw_candle(data, path):
    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis('off')

    mn, mx = data['Low'].min(), data['High'].max()
    norm = lambda x: (x - mn) / (mx - mn + 1e-6)

    for i, r in data.iterrows():
        color = 'green' if r['Close'] >= r['Open'] else 'red'
        ax.plot([i,i],[norm(r['Low']),norm(r['High'])],color='black',lw=1)
        op, cl = norm(r['Open']), norm(r['Close'])
        rect = plt.Rectangle((i-0.3,min(op,cl)),0.6,
                              max(abs(cl-op),0.02),color=color)
        ax.add_patch(rect)

    plt.savefig(path, dpi=64, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# DATA CREATION 
def build_data(start, end, folder, is_train=True):
    raw = yf.download(stocks, start=start, end=end,
                      group_by='ticker', progress=False)
    rows = []

    for s in stocks:
        df = raw[s].dropna().reset_index(drop=True)
        if df.empty:
            continue

        df['Pattern'] = df.apply(check_pattern, axis=1)
        df['Tomorrow'] = df['Close'].shift(-1) - df['Close']
        df['Direction'] = np.where(df['Tomorrow'] > 0, 'Up', 'Down')

        for i in range(20, len(df)-1):
            p = df.loc[i,'Pattern']
            if is_train and p == 'None' and random.random() > 0.15:
                continue

            fname = f"{s}_{i}_{p}.png"
            path = os.path.join(folder, p, fname)

            if not os.path.exists(path):
                draw_candle(df.iloc[i-9:i+1].reset_index(drop=True), path)

            rows.append({
                'path': path,
                'pattern': p,
                'direction': df.loc[i,'Direction'],
                'ret': df.loc[i,'Tomorrow']
            })

    return pd.DataFrame(rows)

# BUILD DATA 
train_df = build_data('2018-01-01','2023-12-31',train_folder,True).reset_index(drop=True)
test_df  = build_data('2024-01-01','2024-06-01',test_folder,False).reset_index(drop=True)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['pattern'],
    random_state=42
)

# IMAGE GENERATORS 
pattern_classes = ['Hammer', 'Doji', 'None']
direction_classes = ['Up', 'Down']
pattern_to_idx = {c: i for i, c in enumerate(pattern_classes)}
direction_to_idx = {c: i for i, c in enumerate(direction_classes)}

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.7,1.3)
)

val_gen = ImageDataGenerator(rescale=1./255)

class TradingSequence(Sequence):
    def __init__(self, df, batch_size, datagen=None, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.datagen = datagen
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

        self.pat_idx = self.df['pattern'].map(pattern_to_idx).values
        self.dir_idx = self.df['direction'].map(direction_to_idx).values

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = self.df.loc[batch_idx, 'path'].values

        x = np.zeros((len(batch_paths), SIZE, SIZE, 3), dtype=np.float32)
        for i, p in enumerate(batch_paths):
            img = load_img(p, target_size=(SIZE, SIZE))
            arr = img_to_array(img)
            if self.datagen is not None:
                arr = self.datagen.random_transform(arr)
                arr = self.datagen.standardize(arr)
            else:
                arr = arr / 255.0
            x[i] = arr

        y_pat = to_categorical(self.pat_idx[batch_idx], num_classes=len(pattern_classes))
        y_dir = to_categorical(self.dir_idx[batch_idx], num_classes=len(direction_classes))

        return x, {'pattern': y_pat, 'direction': y_dir}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_run = TradingSequence(train_df, BATCH, datagen=train_gen, shuffle=True)
val_run   = TradingSequence(val_df, BATCH, datagen=val_gen, shuffle=False)

# MODEL
inp = layers.Input(shape=(SIZE,SIZE,3))

x = layers.Conv2D(32,(3,3),padding='same',activation='relu')(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128,(3,3),padding='same',activation='relu')(x)
x = layers.BatchNormalization()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64,activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)

pattern_out = layers.Dense(3,activation='softmax',name='pattern')(x)
direction_out = layers.Dense(2,activation='softmax',name='direction')(x)

model = models.Model(inp, [pattern_out, direction_out])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={
        'pattern': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        'direction': tf.keras.losses.CategoricalCrossentropy()
    },
    loss_weights={'pattern':1.0,'direction':0.5},
    metrics={'pattern':'accuracy','direction':'accuracy'}
)

# TRAIN
history = model.fit(
    train_run,
    validation_data=val_run,
    epochs=EPOCHS,
    verbose=1
)

# TEST EVALUATION
test_run = TradingSequence(test_df, BATCH, datagen=val_gen, shuffle=False)
pat_pred, dir_pred = model.predict(test_run)

pat_cls = np.argmax(pat_pred, axis=1)
dir_cls = np.argmax(dir_pred, axis=1)

hammer_idx = pattern_to_idx['Hammer']
up_idx = direction_to_idx['Up']

pnl = []
wins = losses = 0

for i,row in test_df.iterrows():
    if pat_cls[i] == hammer_idx and dir_cls[i] == up_idx:
        pnl.append(row['ret'])
        wins += row['ret'] > 0
        losses += row['ret'] <= 0

# RESULTS
print("\n"+"="*40)
print("JOINT LEARNING TRADING RESULTS")
print("="*40)

if wins + losses:
    print(f"Trades     : {wins + losses}")
    print(f"Win Rate   : {(wins/(wins+losses))*100:.2f}%")
    print(f"Total PnL  : {sum(pnl):.2f}")
else:
    print("No trades executed")

print("-"*40)
print(f"Final Train Pattern Acc : {history.history['pattern_accuracy'][-1]*100:.1f}%")
print(f"Final Val   Pattern Acc : {history.history['val_pattern_accuracy'][-1]*100:.1f}%")
print("="*40)

# VISUALIZATIONS
import itertools

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)

# 1) Training history: pattern accuracy & loss
try:
    plt.figure(figsize=(6,4))
    plt.plot(history.history.get('pattern_accuracy', []), label='train_pattern_acc')
    plt.plot(history.history.get('val_pattern_accuracy', []), label='val_pattern_acc')
    if 'direction_accuracy' in history.history:
        plt.plot(history.history.get('direction_accuracy', []), '--', label='train_direction_acc')
        plt.plot(history.history.get('val_direction_accuracy', []), '--', label='val_direction_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'accuracy.png'))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'loss.png'))
    plt.close()
except Exception as e:
    print(f"Visualization error (accuracy/loss): {e}")

# 2) Confusion matrices for pattern and direction on test set
try:
    y_true_pat = test_df['pattern'].map(pattern_to_idx).values
    y_true_dir = test_df['direction'].map(direction_to_idx).values

    def plot_cm(y_true, y_pred, labels, fname, cmap=plt.cm.Blues):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(fname)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir,fname.replace(' ','_') + '.png'))
        plt.close()

    plot_cm(y_true_pat, pat_cls, pattern_classes, 'Pattern Confusion Matrix')
    plot_cm(y_true_dir, dir_cls, direction_classes, 'Direction Confusion Matrix')
except Exception as e:
    print(f"Visualization error (confusion matrices): {e}")

# 3) PnL: histogram
try:
    if len(pnl) > 0:
        cum = np.cumsum(pnl)
        plt.figure(figsize=(6,4))
        plt.plot(cum)
        plt.xlabel('Trade #')
        plt.ylabel('Cumulative PnL')
        plt.title('Cumulative PnL')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir,'cumulative_pnl.png'))
        plt.close()

        plt.figure(figsize=(6,4))
        plt.hist(pnl, bins=30)
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.title('Trade Returns Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir,'pnl_hist.png'))
        plt.close()
except Exception as e:
    print(f"Visualization error (pnl plots): {e}")

# 4) Sample predictions 
try:
    n = min(6, len(test_df))
    sample = test_df.sample(n=n, random_state=42)
    plt.figure(figsize=(12,6))
    for i, idx in enumerate(sample.index):
        img = plt.imread(test_df.loc[idx,'path'])
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.axis('off')
        tpat = pattern_classes[y_true_pat[idx]]
        tdir = direction_classes[y_true_dir[idx]]
        ppat = pattern_classes[pat_cls[idx]]
        pdir = direction_classes[dir_cls[idx]]
        plt.title(f'T:{tpat}/{tdir}\nP:{ppat}/{pdir}', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'sample_predictions.png'))
    plt.close()
