import onnxruntime
import numpy as np
import cv2
import pytesseract


def crop_image(frame, x, y, width, height):
    # Crop the image
    cropped_img = frame[y:y + height, x:x + width]
    return cropped_img


def extractor(frame):
    return pytesseract.image_to_string(frame)


def preprocessor(frame):
    img = cv2.resize(frame, (640, 640))
    image_data = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1] range
    image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    return image_data


class Detector:
    def __init__(self):
        self.model_path = "static_quantized.onnx"
        self.classes = {0: 'Number-plates'}
        self.session = onnxruntime.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.model_inputs = self.session.get_inputs()
        self.input_shape = self.model_inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]

    def draw_detections(self, frame, box, score, class_id):
        x1, y1, w, h = box
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f'{self.classes[class_id]}: {score:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        try:
            cropped = crop_image(frame, x1, y1, w, h)
            text = extractor(cropped)
            print(text)
            cv2.putText(frame, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        finally:
            # cv2.rectangle(frame, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,cv2.FILLED)
            # cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            return frame

    def postprocessor(self, results, frame, confidence, iou):
        img_height, img_width = frame.shape[:2]
        outputs = np.transpose(np.squeeze(results[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        x_factor = img_width / self.input_width  # img_width = 640
        y_factor = img_height / self.input_height  # img_width = 640
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= confidence:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            frame = self.draw_detections(frame, box, score, class_id)
        return frame

    def detector(self, frame):
        ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(frame)
        return self.session.run(["output0"], {"images": ortvalue})

    def pipeline(self, frame):
        frame = self.postprocessor(self.detector(preprocessor(frame)), frame, 0.7, 0.8)
        return frame
