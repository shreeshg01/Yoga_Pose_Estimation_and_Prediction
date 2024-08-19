from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import subprocess

def execute_python_file():
    try:
        # Replace 'your_python_file.py' with the path to your Python file
        subprocess.run(['python', 'test.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    else:
        print("Python file executed successfully.")

# Create the main Tkinter window
root = tk.Tk()
root.title("Execute Python File")

# Set window size to full screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

# Load and display the background image
background_image = Image.open("background_image.jpeg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Add a paragraph above the button
paragraph_label = tk.Label(root, text="Hello... Welcome to Yoga Pose Estimation And Prediction", bg="white", font=("Arial", 24))
paragraph_label.pack(pady=(screen_height * 0.1, screen_height * 0.05))

# Create a button widget
execute_button = tk.Button(root, text="Let's Begin", command=execute_python_file, font=("Arial", 18))
execute_button.pack(pady=(screen_height * 0.05, screen_height * 0.1))

# Run the Tkinter event loop
root.mainloop()
