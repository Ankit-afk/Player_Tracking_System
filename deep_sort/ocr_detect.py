from easyocr import Reader
import cv2

def detect_number(roi):
    """
    This is a test function not used in player identification system. It was implemented for testing purposes.

    This function is used to detect and identify text and number in the region of interest (roi) parameter.
    Returns the probability along with the detected text.
    """

    resized_image = cv2.resize(roi, (500, 500)) 
    reader = Reader(["en"], gpu=True)
    # reader = Reader(["en"], gpu=False)
    results = reader.readtext(resized_image, allowlist="0123456789")
    if not results:
        print("No numbers detected.")
        return None
    else: 
        text = results[0][1]    #string 
        # prob = results[0][2]
        return text
        # print("[INFO] {:.4f}: {}".format(prob, text))

# image = cv2.imread("images2/p1.jpg")
# text = detect_number(image)
# print("Found :",text)