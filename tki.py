from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
from keras.models import load_model
import os


def fun():
    model = load_model("model.h5")

    classes = {
        4: ("nv", " melanocytic nevi"),
        6: ("mel", "melanoma"),
        2: ("bkl", "benign keratosis-like lesions"),
        1: ("bcc", " basal cell carcinoma"),
        5: ("vasc", " pyogenic granulomas and hemorrhage"),
        0: ("akiec", "Actinic keratoses and intraepithelial carcinomae"),
        3: ("df", "dermatofibroma"),
    }
    srcdir = ".\\upload"
    count = 0
    for temp in os.listdir(srcdir):
        img = cv2.imread(os.path.join(srcdir, temp))
        cv2.imwrite(temp, img)
        img = cv2.resize(img, (28, 28))
        result = model.predict(img.reshape(1, 28, 28, 3))
        max_prob = max(result[0])
        class_ind = list(result[0]).index(max_prob)
        class_name = classes[class_ind]
        print(class_name)
        count += 1
        if count > 2:
            break
        return class_name


root = Tk()
root.geometry("900x900")


def save_image():
    # Open the file dialog box to select the image
    filename = filedialog.askopenfilename(
        title="Select Image",
        filetypes=(
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*"),
        ),
    )
    # Open the image using Pillow
    image = Image.open(filename)

    # Save the image to a folder
    save_location = filedialog.asksaveasfilename(
        initialdir="/", defaultextension=".png", filetypes=(("PNG files", "*.png"),)
    )
    image.save(save_location)
    image = Image.open(filename)

    # Display the image in a tkinter window
    photo = ImageTk.PhotoImage(image)
    label = Label(root, image=photo)
    label.image = photo
    label.pack()
    ans = fun()
    text = Label(root, text=ans)
    text.place(x=700, y=700)


# Create a button to call the save_image function
save_button = Button(root, text="Save Image", command=save_image)
save_button.pack()


root.mainloop()
