# Run: python3 Predict.py
# ----Thu vien----
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time

# --- Load anh nhan dien (gray) ----
name = "so5.png"
image = cv2.imread("images/"+name)

mnist = False
if mnist:
    # ---Load model va trong so ----
    model = load_model("weights/model_sudoku_mac.weights.h5")
    model.load_weights("weights/weight_sudoku_mac.weights.h5")
    # Chuyen ve anh xam
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Reseze ve cung kich thuoc voi tap train
    image_resized = cv2.resize(image_gray,(28,28))
    # Thay doi background
    image_convert = 255 - image_resized
    # Thay doi shape va Chuyen scale ve [0,1]
    image_array = image_convert.reshape(1,28,28,1).astype('float32')/255

else:
    model = load_model("weights/digits_weight_10.h5")
    # Resize anh 
    image_resized = cv2.resize(image,(32,32))
    # Anh xam
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # Chuyen ve anh binary
    image_binary = cv2.threshold(image_gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Thay doi shape va Chuyen scale ve [0,1]
    image_array = image_binary.reshape(1,32,32,1).astype("float32")/255.0


# ----Nhan dang-----
# Xac suat cua doi tuong nhan dang
t1 = time.time()
scores = model.predict(image_array)
# Lay doi tuong co xac suat lon nhat
id_class = np.argmax(scores)
score = scores[0][id_class]
t2 = time.time()
# Thoi gian nhan dang
print("Time: {:.4f}".format(t2-t1))

# ---Hien thi va luu-----
# Resize anh
image = cv2.resize(image,(128,128)) 
# Ghi chu thich
txt = "So {}-{:.2f}%".format(id_class,float(score)*100)
cv2.putText(image, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
    0.5, (0, 0, 255), 1)
    
# Luu anh 
#outPath = "out_images/mydata"+name
#cv2.imwrite(outPath,image)
# Hien thi anh

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
