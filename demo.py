import cv2
from transformers import ViltProcessor, ViltForQuestionAnswering

# Define the ViltForQuestionAnswering pipeline
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Open the camera
cap = cv2.VideoCapture(0)

# Set the size of the window
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 800, 600)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:    

        
        # Check if a key was pressed
        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key was pressed, exit the loop
        if key == ord('q'):
            break

        # If the 'enter' key was pressed, ask the question
        elif key == 13:
            # Wait for the user to enter a question
            question = input("Enter a question: ")
            
            # prepare inputs
            encoding = processor(frame, question, return_tensors="pt")

            # forward pass
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            
            print("Predicted answer:", model.config.id2label[idx])

    else:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
