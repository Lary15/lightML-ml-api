import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  for metric in hist.columns:
      if metric != 'epoch' and "val" not in metric:
          plt.figure()
          plt.xlabel('epoch')
          plt.ylabel(metric)
          plt.plot(hist['epoch'], hist[metric])
          plt.plot(hist['epoch'], hist[f"val_{metric}"])
          plt.legend(['train', 'val'], loc='upper left')

  plt.show()

# Load the csv files
train_df = pd.read_csv('../lightML-AI/new_train.csv')
test_df = pd.read_csv('../lightML-AI/new_test.csv')
val_df = pd.read_csv('../lightML-AI/new_val.csv')

train_y = train_df.pop('led_on')
val_y = val_df.pop('led_on')
test_y = test_df.pop('led_on')

# Add a timestamp column
os.environ['TZ'] = 'Brazil/East'
time.tzset()

train_df["weekday"] = train_df["timestamp"].apply(lambda a: int(time.strftime('%w', time.localtime(a))))
val_df["weekday"] = val_df["timestamp"].apply(lambda a: int(time.strftime('%w', time.localtime(a))))
test_df["weekday"] = test_df["timestamp"].apply(lambda a: int(time.strftime('%w', time.localtime(a))))

train_df["timestamp"] = train_df["timestamp"].apply(lambda a: int(time.strftime('%H', time.localtime(a))))
val_df["timestamp"] = val_df["timestamp"].apply(lambda a: int(time.strftime('%H', time.localtime(a))))
test_df["timestamp"] = test_df["timestamp"].apply(lambda a: int(time.strftime('%H', time.localtime(a))))

for df in [train_df, val_df, test_df]:
    df['mic'] = (df['mic']-0)/(4096-0)
    df['temp'] = (df['temp']-0)/(45-0)
    df['hum'] = (df['hum']-0)/(100-0)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                        tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()],
)

# Train the model
history = model.fit(train_df.to_numpy(), train_y.to_numpy(), epochs=200, validation_data=(val_df.to_numpy(), val_y.to_numpy()))
plot_history(history)

# Evaluate the model
model.evaluate(test_df.to_numpy(), test_y.to_numpy())

# Save the model
tf.saved_model.save(model, "dnn")