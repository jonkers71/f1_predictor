import os
import matplotlib.pyplot as plt
from datetime import datetime

def save_plot(fig, title: str):
    os.makedirs("visuals", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"visuals/{title}_{timestamp}.png"
    fig.savefig(filename)
    plt.close(fig)
    return filename
