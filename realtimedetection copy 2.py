import cv2
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load the pre-trained model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("emotiondetector.h5")

# Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Fuzzy logic setup
brightness = ctrl.Antecedent(np.arange(0, 256, 1), 'brightness')
brightness['dark'] = fuzz.trimf(brightness.universe, [0, 0, 80])  # Reduced max for 'dark'
brightness['medium'] = fuzz.trimf(brightness.universe, [40, 80, 160])  # Shifted 'medium' to lower range
brightness['bright'] = fuzz.trimf(brightness.universe, [100, 200, 255])  # Reduced min for 'bright'

emotion_confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'emotion_confidence')
emotion_confidence['low'] = fuzz.trimf(emotion_confidence.universe, [0, 0, 0.3])
emotion_confidence['medium'] = fuzz.trimf(emotion_confidence.universe, [0.2, 0.5, 0.8])
emotion_confidence['high'] = fuzz.trimf(emotion_confidence.universe, [0.7, 1, 1])

final_decision = ctrl.Consequent(np.arange(0, 11, 1), 'final_decision')
final_decision['reject'] = fuzz.trimf(final_decision.universe, [0, 1, 2])
final_decision['uncertain'] = fuzz.trimf(final_decision.universe, [3, 5, 7])
final_decision['accept'] = fuzz.trimf(final_decision.universe, [7, 8, 10])

# Define fuzzy rules
rule_accept = ctrl.Rule(emotion_confidence['high'] & brightness['bright'], final_decision['accept'])
rule_uncertain = ctrl.Rule(emotion_confidence['medium'] | brightness['medium'], final_decision['uncertain'])
rule_reject = ctrl.Rule(emotion_confidence['low'] | brightness['dark'], final_decision['reject'])

# Fuzzy control system
decision_ctrl = ctrl.ControlSystem([rule_accept, rule_uncertain, rule_reject])
decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)

# Bayesian prediction confidence function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Webcam input and fuzzy decision integration
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break
    
    # Convert to HSV for brightness calculation
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]  # The Value (brightness) channel
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (p, q, r, s) in faces:
        # Preprocess face image
        face_image = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)

        # Bayesian inference
        pred = model.predict(img)[0]  # Get probabilities
        priors = np.array([1/7] * 7)  # Assuming uniform priors
        posteriors = priors * pred
        posteriors /= posteriors.sum()
        prediction_label = labels[np.argmax(posteriors)]
        confidence_score = posteriors.max()

        # Image brightness calculation
        image_brightness = np.mean(value_channel)  # Use mean of the Value channel

        # Fuzzy system input
        decision_sim.input['emotion_confidence'] = confidence_score
        decision_sim.input['brightness'] = image_brightness
        decision_sim.compute()

        # Final decision output
        fuzzy_output = decision_sim.output['final_decision']
        decision_category = (
            "Accept" if fuzzy_output >= 7 else 
            "Uncertain" if fuzzy_output >= 3 else 
            "Reject"
        )

        # Display outputs on the screen
        cv2.putText(im, f'{prediction_label} ({confidence_score:.2f})', (p - 10, q - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.putText(im, f'Brightness: {image_brightness:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0))
        cv2.putText(im, f'Decision: {decision_category}', (10, 60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
    
    cv2.imshow("Output", im)
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
