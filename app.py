from flask import Flask, render_template, Response, jsonify, session
from datetime import datetime
import os
import cv2
import math
from ultralytics import YOLO
import json
import time
import Adafruit_ADS1x15
import RPi.GPIO as GPIO
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize Flask app and set a secret key
app = Flask(__name__)
app.config['SECRET_KEY'] = 'innotechdomain'
class_name = ""
x_val = 0
y_val = 0
z_val = 0

# Function to monitor earthquake
def monitor_earthquake():

    # Create I2C bus
    i2c = busio.I2C(board.SCL, board.SDA)

    # Define the ADC and buzzer pin
    # Create the ADC object
    ads = ADS.ADS1115(i2c)
    BUZZER_PIN = 24
    LED_PIN = 23

    # Define accelerometer pins
    X_PIN = 0
    Y_PIN = 1
    Z_PIN = 2

    # Variables
    SAMPLES = 50
    MAX_VAL = 600  # max change limit
    MIN_VAL = -60  # min change limit
    BUZZ_TIME = 1  # buzzer on time in seconds
    # Create analog input objects
    x_analog_in = AnalogIn(ads, getattr(ADS, f'P{X_PIN}'))
    y_analog_in = AnalogIn(ads, getattr(ADS, f'P{Y_PIN}'))
    z_analog_in = AnalogIn(ads, getattr(ADS, f'P{Z_PIN}'))

    # Calibration
    x_sample = sum(x_analog_in.value for _ in range(SAMPLES)) / SAMPLES
    y_sample = sum(y_analog_in.value for _ in range(SAMPLES)) / SAMPLES
    z_sample = sum(z_analog_in.value for _ in range(SAMPLES)) / SAMPLES

    # Setup GPIO pins
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)
    buz = 0

    print(" X Y Z ")

    global x_val
    global y_val
    global z_val

    # Data for plotting
    x_values = []
    y_values = []
    z_values = []

    # Main loop
    try:
        while True:
            # Recalibrate within the loop
            x_sample = sum(x_analog_in.value for _ in range(SAMPLES)) / SAMPLES
            y_sample = sum(y_analog_in.value for _ in range(SAMPLES)) / SAMPLES
            z_sample = sum(z_analog_in.value for _ in range(SAMPLES)) / SAMPLES

            value_x = x_sample - x_analog_in.value
            value_y = y_sample - y_analog_in.value
            value_z = z_sample - z_analog_in.value

            if (
                value_x < MIN_VAL
                or value_x > MAX_VAL
                or value_y < MIN_VAL
                or value_y > MAX_VAL
                or value_z < MIN_VAL
                or value_z > MAX_VAL
            ):
                if buz == 0:
                    start = time.time()
                buz = 1
            elif buz == 1:
                print("Earthquake Alert")
                if time.time() >= start + BUZZ_TIME:
                    buz = 0
            else:
                print(" X Y Z ")

            # Update plot data
            x_values.append(value_x)
            y_values.append(value_y)
            z_values.append(value_z)

            #global variables for detection
            x_val = value_x
            y_val = value_y
            z_val = value_z

            # GPIO output
            GPIO.output(BUZZER_PIN, buz)
            GPIO.output(LED_PIN, buz)

    # Handle KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()

# Function for video detection using YOLO
def video_detection(path_x):

    GPIO.setmode(GPIO.BCM)
    buzzer_pin = 19
    led_pin = 25
    GPIO.setup(buzzer_pin, GPIO.OUT)
    GPIO.setup(led_pin, GPIO.OUT)

    # Flag to track detection status
    detection_flag = False

    # Initialize video capture
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    global class_name
    # Load YOLO model and class names
    model = YOLO("yolov8mfire.pt")
    classNames = ['smoke', 'fire']

    # Loop through video frames
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # Reset detection flag
        detection_flag = False

        # Loop through detected objects
        for r in results:
            boxes = r.boxes
            # check if there's boxes detected; empty the class_name
            if len(boxes) == 0:
                class_name = ""
            else:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    label = f'{class_name}{conf}'
                    t_size = cv2.getTextSize(
                        label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    color = (0, 204, 255) if class_name == '' else (
                        85, 45, 255)
                    if conf > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.rectangle(img, (x1, y1), c2, color, -
                                      1, cv2.LINE_AA)  # filled
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1,
                                    [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    detection_flag = True
        yield img
        # Control the buzzer and LED based on detection flag
        if detection_flag:
            # Turn on buzzer and LED
            GPIO.output(buzzer_pin, GPIO.HIGH)
            GPIO.output(led_pin, GPIO.HIGH)
        else:
            # Turn off buzzer and LED
            GPIO.output(buzzer_pin, GPIO.LOW)
            GPIO.output(led_pin, GPIO.LOW)

    # Cleanup GPIO
    GPIO.cleanup()
# Function to generate video frames
def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to generate video frames for web display
def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define routes for the Flask app
@app.route('/', methods=['GET', 'POST'])
def index():
    session.clear()
    return render_template('indexnew.html')


@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('indexnew.html')


@app.route('/start_detection', methods=['GET', 'POST'])
def start_detection():
    # Set a session variable to indicate that the detection has been started
    session['detection_started'] = True
    return jsonify({'message': 'Detection started successfully'})


@app.route('/earthquake_detection_started', methods=['GET', 'POST'])
def earthquake_detection_started():
    # Set a session variable to indicate that the detection has been started
    session['detection_started'] = True
    monitor_earthquake()

# Function to detect fire or smoke
def detect_fire_smoke():
    if class_name == 'fire':
        return "Fire detected!"
    elif class_name == 'smoke':
        return "Smoke detected!"
    elif class_name == 'fire' and class_name == 'smoke':
        return "Fire and Smoke detected!"
    else:
        return "No fire or smoke detected."

# Function to detect earthquake
def detect_earthquake():
    # Call the detect_earthquake function
    max_val = 60  # max change limit
    min_val = -60

    if (x_val < min_val or x_val > max_val or y_val < min_val or y_val > max_val or z_val < min_val or z_val > max_val):
        return "Earthquake detected!"
    else:
        return "No earthquake detected."


@app.route('/detection_status', methods=['GET'])
def detection_status():
    status_message = detect_fire_smoke()

    # Log fire detection to the DataTable and save to text file
    if "Fire detected!" in status_message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'status_message': status_message})

# Route to pass the value from web
@app.route('/earthquake_status', methods=['GET'])
def earthquake_status():
    status_message = detect_earthquake()

    if "Earthquake detected!" in status_message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'status_message': status_message})


@app.route('/xyz_values', methods=['GET'])
def xyz_values():
    """Returns a JSON object containing the XYZ values from the accelerometer"""
    return jsonify({'x_value': x_val, 'y_value': y_val, 'z_value': z_val})


# To display the Output Video on the Webcam page
@app.route('/webapp')
def webapp():
    # Check if the detection has been started before streaming frames
    if session.get('detection_started'):
        return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'message': 'Detection not started yet'})

# Run the Flask app
if __name__ == "__main__":
    cv2.destroyAllWindows()
    app.run(debug=True, host='192.168.100.104', port='8000')