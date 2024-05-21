# Run: python3 SolvingSudoku_Mydata.py --image images/1.png --output results/1.png
from tensorflow.keras.models import load_model # type: ignore
from tensorflow import keras
from sudoku import Sudoku
import numpy as np
import cv2
import os


DEBUG = False

def pre_data(image_path: str):
    """
    Input: image path
    Return: image binary, image resize, height, width have resize
    """
    image = cv2.imread(image_path)
    image_resize = cv2.resize(image,(360,360))
    height,width = image_resize.shape[:2]

    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if DEBUG:
        cv2.imshow("Original Binary Image",image_binary)

    return (height, width, image_resize, image_binary)

def predict_solve(model, height,width,image_binary):
    """
    Input: model, image binary
    Return: board have prediction.
    """

    container = [[0 for _ in range(9)] for _ in range(9)]
    template = [[0 for _ in range(9)] for _ in range(9)]
    board = np.zeros((9, 9), dtype="int")

    small_h = height//9
    small_w = width//9

    print(height,width)
    for i in range(9):
        for j in range(9):
            image_crop = image_binary[i*small_h:(i+1)*small_h,j*small_w:(j+1)*small_w]
            # Xoa vien anh
            image_crop =image_crop[3:small_h-3,3:small_w-3]
            # Resize anh 
            image_crop = cv2.resize(image_crop,(32,32))

            if DEBUG:
                cv2.imshow("Digit",image_crop)
                cv2.waitKey(100)
            if DEBUG:
                print(np.mean(image_crop))

            if np.mean(image_crop) <250:
                # Conver to [0,1]
                image_array = image_crop.reshape(1,32,32,1).astype("float32")/255.0

                predictions = model.predict(image_array)  # x_test is your test data
                predicted_classes = np.argmax(predictions, axis=1)
                
                if DEBUG:
                    print("Debug",predicted_classes)

                container[i][j] = int(predicted_classes)
                template[i][j] = int(predicted_classes)
                board[i, j] = int(predicted_classes)

    print("[INFO] OCR'd sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    print(board)
    puzzle.show() 

    print("[INFO] solving sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()    
    return solution, template
    
def display_result(solution,template,image):
    cv2.imshow("Original", image)
    height,width = image.shape[:2]
    small_h = height//9
    small_w = width//9
    # Draw result
    for i in range(9):
        for j in range(9):
            if template[i][j] == 0:
                txt = str(solution.board[i][j])
                x = j*small_w+10
                y = i*small_h+30
                cv2.putText(image, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)

    # Draw line
    for i in range(0,4):
        cv2.line(image,(i*int(height/3),0),(i*int(height/3),height),(0,0,255),3)
    for i in range(0,4):
        cv2.line(image,(0,i*int(width/3)),(width,i*int(width/3)),(0,0,255),3)
    
    cv2.imshow("Solve Sudoku", image)

def main() -> None:
    model = load_model("weights/digits_weight_10_mac.h5")
    image_path = "/Users/tuantran/Desktop/Sudoku/images/1.png"
    images_path = "/Users/tuantran/Desktop/Sudoku/images/"
    
    for file_name in os.listdir(images_path):
        # Check if the file has a .png extension
        if file_name.endswith('.png'):
            # Construct the full file path and append to the list
            image_path = os.path.join(images_path, file_name)
            
            (height, width, image_resize, image_binary) = pre_data(image_path)
            solution, template = predict_solve(model=model, height=height, width=width,image_binary=image_binary)

            display_result(solution=solution,template=template,image=image_resize)

            cv2.waitKey(30)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()