import tkinter as tk
from tkinter import filedialog
from inference import predict_digit
from PIL import Image, ImageTk, ImageOps  # Import Pillow modules
import io 


def select_image():
    global image_label
    file_path = filedialog.askopenfilename()
    if file_path:
        # Get the predicted digit
        digit = predict_digit(file_path)
        label.config(text=f"Predicted Digit: {digit}")

        # Load the image using Pillow
        pil_image = Image.open(file_path)
        # Optionally resize the image to fit your UI
        pil_image.thumbnail((400, 400))  # This resizes the image while keeping the aspect ratio
        
        # Convert the PIL image to a Tkinter-compatible image
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update the image_label to display the new image
        image_label.config(image=tk_image)
        image_label.image = tk_image  # Save a reference to prevent garbage collection

def predict_canvas_drawing():
    global image_label
    ps = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    img = ImageOps.grayscale(img)

    img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    temp_filename = "canvas_digit.png"
    img.save(temp_filename)
    
    digit = predict_digit(temp_filename)
    label.config(text=f"Predicted Digit: {digit}")

    display_img = img.resize((200, 200), Image.NEAREST)
    tk_image = ImageTk.PhotoImage(display_img)
    image_label.config(image=tk_image)
    image_label.image = tk_image

def clear_canvas():
    canvas.delete("all")


# Create the main window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("800x650")

# Button to upload image
btn = tk.Button(root, text="Upload image", command=select_image)
btn.pack()

# Label to display prediction result
label = tk.Label(root, text="Prediction will appear here")
label.config(font=("Helvetica", 12, "bold"), fg="red")
label.pack()

# Label to display the image
image_label = tk.Label(root)
image_label.pack()

canvas_frame = tk.Frame(root)
canvas_frame.pack(pady=10)

canvas_width, canvas_height = 300, 300
canvas = tk.Canvas(canvas_frame, width=canvas_width, heigh=canvas_height, bg="White")
canvas.pack()



def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")

canvas.bind("<B1-Motion>", paint)

btn_predict_canvas = tk.Button(root, text="Predict from Drawing", command=predict_canvas_drawing)
btn_predict_canvas.pack(pady=5)

btn_clear = tk.Button(root, text="Clear Canvas", command=clear_canvas)
btn_clear.pack(pady=5)

root.mainloop()
