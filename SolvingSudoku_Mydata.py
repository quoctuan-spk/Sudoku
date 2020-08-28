# Run: python3 SolvingSudoku_Mydata.py --image images/1.png --output results/1.png
# --- Thu vien---
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import argparse

# ----Xay dung argument parse-----
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-o", "--output", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# --- Load model ---
model = load_model("weights/digits_weight_10.h5")

# ---- Load anh ---
image = cv2.imread(args["image"])
image = cv2.resize(image,(360,360))
h,w = image.shape[:2]
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_binary = cv2.threshold(image_gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow("Anh goc",image_binary)

debug = False

# ----Split anh va nhan dang ----
# Khai bao template container 
container = [[0 for _ in range(9)] for _ in range(9)]
template = [[0 for _ in range(9)] for _ in range(9)]
board = np.zeros((9, 9), dtype="int")

small_h = h//9
small_w = w//9
# ----Nhan dang chu so----
for i in range(9):
    for j in range(9):
        # Crop anh 
        image_crop = image_binary[i*small_h:(i+1)*small_h,j*small_w:(j+1)*small_w]
        # Xoa vien anh
        image_crop =image_crop[3:small_h-3,3:small_w-3]
        # Resize anh 
        image_crop = cv2.resize(image_crop,(32,32))

        if debug:
            cv2.imshow("Digit",image_crop)
            cv2.waitKey(20)

        # Kiem tra co doi tuong se nhan dang
        # print(np.mean(image_crop))
        if np.mean(image_crop) <250:
            # Chuyen anh ve [0,1]
            image_array = image_crop.reshape(1,32,32,1).astype("float32")/255.0
            # Nhan dang
            container[i][j] = int(model.predict_classes(image_array))
            template[i][j] = int(model.predict_classes(image_array))

            board[i, j] = int(model.predict_classes(image_array))

# ----Giai----
print("[INFO] OCR'd sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

# solve the sudoku puzzle
print("[INFO] solving sudoku puzzle...")
solution = puzzle.solve()
solution.show_full()

solving = False
while solving:
    container, stump_count = explicit_solver(container)
    zero_count = 0
    for l in container:
        for v in l:
            if v == 0:
                zero_count += 1
    if zero_count==0:
        solving=False
    if stump_count > 0:
        for i in range(9):
            for j in range(9):
                container = implicit_solver(i,j,container)

# --- Dien so -----
for i in range(9):
    for j in range(9):
        if template[i][j] == 0:
            txt = str(solution.board[i][j])
            x = j*small_w+10
            y = i*small_h+30
            cv2.putText(image, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
        
# ---- Ve line ----
for i in range(0,4):
    cv2.line(image,(i*int(h/3),0),(i*int(h/3),h),(0,0,255),3)
for i in range(0,4):
    cv2.line(image,(0,i*int(w/3)),(w,i*int(w/3)),(0,0,255),3)
    
# --- Save ---
cv2.imwrite(args["output"],image)

# --- Hien thi------
cv2.imshow("Result",image)

cv2.waitKey(0)
cv2.destroyAllWindows()