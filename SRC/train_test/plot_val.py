"""

    Machine Learning Project Work: Tennis Table Tournament
    Group 2:
        Ciaravola Giosuè - g.ciaravola3@studenti.unisa.it
        Conato Christian - c.conato@studenti.unisa.it
        Del Gaudio Nunzio - n.delgaudio5@studenti.unisa.it
        Garofalo Mariachiara - m.garofalo38@studenti.unisa.it

    ---------------------------------------------------------------

    plot_val.py

    A file useful for creating a plot to show the loss trend based
    on the report (.csv) that records the loss values during the
    epochs of supervised training.

"""

import pandas as pd
import matplotlib.pyplot as plt

# Read csv report
df = pd.read_csv('arm_learning_report.csv')

# Plot creation
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['training_loss'], label='Training Loss', color='b')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('plot.png')  # Sostituisci con il percorso desiderato

plt.close()