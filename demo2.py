import cv2
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch 


# Define the ViltForQuestionAnswering pipeline
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Define function to capture live video using OpenCV
def capture_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Live Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_answer(frame):
        question = input("Enter your question: ")
        #pixel_values = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        encoding = processor(frame, question, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer

def display_answer(frame, answer):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(frame, answer, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow('Live Video', frame)
    cv2.waitKey(3000)

while True:
    capture_video()
    
    # Get answer and display on the video frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        answer = get_answer(frame)
        display_answer(frame, answer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()