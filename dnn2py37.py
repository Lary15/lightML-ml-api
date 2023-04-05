import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

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
train_df = pd.read_csv('../lightML-AI/train.csv')
test_df = pd.read_csv('../lightML-AI/test.csv')
val_df = pd.read_csv('../lightML-AI/val.csv')

train_y = train_df.pop('led_on')
val_y = val_df.pop('led_on')
test_y = test_df.pop('led_on')

# Normalize
for col in ["temp", "hum", "mic"]:
    train_df[col] = (train_df[col]-train_df[col].min())/(train_df[col].max()-train_df[col].min())
    val_df[col] = (val_df[col]-val_df[col].min())/(val_df[col].max()-val_df[col].min())
    test_df[col] = (test_df[col]-test_df[col].min())/(test_df[col].max()-test_df[col].min())


# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(35, activation='relu'),
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(42, activation='relu'),
    tf.keras.layers.Dense(49, activation='relu'),
    tf.keras.layers.Dense(49, activation='relu'),
    tf.keras.layers.Dense(21, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                        tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()],
)

# Train the model
history = model.fit(train_df.to_numpy(), train_y.to_numpy(), epochs=250, validation_data=(val_df.to_numpy(), val_y.to_numpy()))
plot_history(history)

# Evaluate the model
model.evaluate(test_df.to_numpy(), test_y.to_numpy())

# Save the model
tf.saved_model.save(model, "dnn")