import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pygame


class GestureRecogniser():
    def __init__(self, model_path):
      self.model_path = model_path
      self.gesture_text = ""
      self.gesture_name = ""
      self.gesture_timestamp = 0
      #setting up gesture object
      self.BaseOptions = mp.tasks.BaseOptions
      self.GestureRecognizer = vision.GestureRecognizer
      self.GestureRecognizerOptions = vision.GestureRecognizerOptions
      #self.GestureRecognizerResult = vision.GestureRecognizerResult
      self.VisionRunningMode = mp.tasks.vision.RunningMode
      #enabling the use of api
      options = self.GestureRecognizerOptions(base_options = self.BaseOptions(model_asset_path=model_path),
                                               running_mode=self.VisionRunningMode.LIVE_STREAM,
                                                 result_callback=self.identifyGesture)
      self.recognizer = self.GestureRecognizer.create_from_options(options)
      # array of explosion images
      explosion_paths = [
        "explosion1.png",
        "explosion2.png",
        "explosion3.png",
        "explosion4.png",
        "explosion5.png"
      ]
      self.explosion_images = []
      for path in explosion_paths:
        img = cv2.imread(path)
        if img is None:
          print(f"Image is none: {path}")
        else:
          print(f"Image loaded: {path}")
        self.explosion_images.append(img)
      #changing colour and size of images
      self.explosion_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.explosion_images]
      self.explosion_images = [cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA) for img in self.explosion_images]
      self.explosion_index  = 0
      self.explosion_active = False
      self.explosion_position = None
      self.explosion_start = None

    def identifyGesture(self, result, output_image: mp.Image, timestamp_ms: int):
       pygame.mixer.init() #sets up sound with pygame
       exploadSound = pygame.mixer.Sound("explosion.wav") #stores sound effect
       if result.gestures:
        #get confidence of gesture
        self.gesture_name = result.gestures[0][0].category_name
        confidence = result.gestures[0][0].score
        #print(result)
        #gesture detection for name displaying and explosion logic
        if confidence > 0.5 and self.gesture_name:
            self.gesture_text = f"Gesture: {self.gesture_name}"
            self.gesture_timestamp = time.time()
            print(self.gesture_text)
            #detect transition for explosion flag
            if self.gesture_name.lower() == "open_palm":
              print("Open Palm is happenning")
              self.explosion_active = True
              self.explosion_start = time.time()
              if not pygame.mixer.get_busy():#makes sure sound isnt already playing
                exploadSound.play()#plays sound

    def palmPosition(self, hand_landmarks):
      # used to detect the centre of the palm so explosion can be placed
      wrist = hand_landmarks.landmark[0]
      palm = hand_landmarks.landmark[9]
      middle_finger = hand_landmarks.landmark[12]
      x = (wrist.x + palm.x + middle_finger.x) / 3 # calculates average between positions
      y = (wrist.y + palm.y + middle_finger.y) / 3
      #print(f"Palm position: {x, y}")
      return(x, y)
    
    def explosion(self, image):
      #runs the explosion logic
      if self.explosion_active:
        #print("Explosion is active") #debug
        elapsed_time = time.time() - self.explosion_start
        frame_duration = 0.1
        #print(f"Elapsed time: {elapsed_time}")
        #print(f"Below if check calc: {self.explosion_index * frame_duration}") # debug
        if elapsed_time > frame_duration:
          print(f"Explosion Frame: {self.explosion_index}") #debug
          image = self.overlayExplosion(image)
          self.explosion_index += 1
          self.explosion_start = time.time()
          if self.explosion_index >= len(self.explosion_images):
            self.explosion_active = False
            self.explosion_index = 0
    
    def overlayExplosion(self, image):
      # function houses logic for overlaying explosion on camera frame
      if self.explosion_position and self.explosion_active:
        print(f"Explosion Position: {self.explosion_position}") #debug
        h, w, _ = image.shape
        x = int(self.explosion_position[0] * w)
        y = int(self.explosion_position[1] * h)
        explosion_frame = self.explosion_images[self.explosion_index]
        if explosion_frame is None:
          print("Explosion frame is none") # debug
          return image
        ex_h, ex_w, _ = explosion_frame.shape
        print(f"Explosion Frame Size: {explosion_frame.shape}") #debug
        # making sure explosion stays within frame
        if x + ex_w > w:
          x = w - ex_w
        if y + ex_h > h:
          y = h - ex_h
        #place explosion frame
        image[y:y+ex_h, x:x+ex_w] = explosion_frame
      return image

class FaceRecogniser():
    def __init__(self):
      self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
      self.initial_nose = None
      self.face_gesture = ""
      self.gesture_timestamp_face = 0

    def detect_movement(self, image):
      pygame.mixer.init() #sets up sound with pygame
      talking_sound = pygame.mixer.Sound("talking.wav") #stores sound effect
      results = self.face_mesh.process(image)
      if not results.multi_face_landmarks:
        return None
      face_landmarks = results.multi_face_landmarks[0] # get face
      # to detect mouth moving
      top_lip = face_landmarks.landmark[13] # id of landmark matches face position
      bottom_lip = face_landmarks.landmark[14]
      lip_distance = abs(top_lip.y - bottom_lip.y)
      if lip_distance > 0.02:
        self.face_gesture = "Talking!"
        self.gesture_timestamp_face = time.time()
        if not pygame.mixer.get_busy():#makes sure sound isnt already playing
          talking_sound.play()#plays sound
      #to detect head shaking
      current_nose = face_landmarks.landmark[1].x
      if self.initial_nose is not None and abs(current_nose - self.initial_nose) > 0.05:
        self.face_gesture = "Head shaking!"
        self.gesture_timestamp_face = time.time()
      self.initial_nose = current_nose

class drawingApp():
    def __init__(self):
      self.colour_palette = [
          (50, 50, 20, (0,0,255)),
          (150, 50, 20, (0,255,0)),
          (250, 50, 20, (255,0,0)),
          (350, 50, 20, (0,255,255)),
          (450, 50, 20, (200,0,150))
      ]
      self.selected_colour = (255,255,255)
      self.drawing = False
      self.prev_position = None
    
    def draw_palette(self, image):
      for x, y, r, colour in self.colour_palette:
        cv2.circle(image, (x, y), r, colour, -1)
    
    def select_color(self, x, y):
      for cx, cy, r, colour in self.colour_palette:
        print(f"Coordinates: {x}, {y} vs circle: {cx}, {cy}")
        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
          print(f"changed colour: {self.selected_colour}")
          self.selected_colour = colour
          return True
      return False
    
    def draw(self, image, x, y):
      if self.drawing:
        cv2.circle(image, (x, y), 10, self.selected_colour, -1)
      self.prev_position = (x, y)

class HandFaceTrackApp():
    def __init__(self, model_path, webcam_id=0):
      print("Initialising webcam...")
      self.cap = cv2.VideoCapture(webcam_id)
      if not self.cap.isOpened():
        print("Error: Could not open video capture")
        return
      print("Webcam initialised")
      self.gesture_recognizer_handler = GestureRecogniser(model_path)
      self.mp_hands = mp.solutions.hands
      self.face_recogniser_handler = FaceRecogniser()
      self.mp_face_detection = mp.solutions.face_detection
      self.drawingApp = drawingApp()
      self.mp_drawing = mp.solutions.drawing_utils
      self.mp_drawing_styles = mp.solutions.drawing_styles
      self.model_path = model_path

    def frame(self, image):
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return image
    
    def displayText(self, image):
      try:
        if image is not None:
          #print(f"Image properties: shape={image.shape}, dtype={image.dtype}, size={image.size}")
          if image.shape[0] > 0 and image.shape[1] > 0:
            if time.time() - self.gesture_recognizer_handler.gesture_timestamp < 2:   
                #display text for which gesture
                cv2.rectangle(image, (50, 370), (400, 410), (19, 69, 139), 50)
                cv2.putText(image, self.gesture_recognizer_handler.gesture_text, (50, 400),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            if time.time() - self.face_recogniser_handler.gesture_timestamp_face < 2: # display for face
                cv2.rectangle(image, (400, 370), (620, 410), (19, 69, 139), 50)
                cv2.putText(image, (self.face_recogniser_handler.face_gesture), (390,400), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
          print("Error: Invalid image passed to displayText")
      except Exception as e:
         print(f"Error in displayText: {e}")
      return image
    
    def run(self): # here the with section starts the hand and face drawing for detection
      with self.mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, \
          self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while self.cap.isOpened():
          #print("Attempting capture")
          success, image = self.cap.read()
          if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
          #print("Captured frame successful")
          if image is None or image.size == 0:
            print("Error: Empty frame")
            break

          # To improve performance, optionally mark the image as not writeable to
          # pass by reference.
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          hand_results = hands.process(image) # passing results individually for processing of hand and face 
          face_results = face_detection.process(image)
          #get frame and call function
          mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
          self.gesture_recognizer_handler.recognizer.recognize_async(mp_image,
                                                                      timestamp_ms=cv2.getTickCount())
          #call face function
          self.face_recogniser_handler.detect_movement(image)
          #call explosion function
          self.gesture_recognizer_handler.explosion(image)
          # Draw the hand annotations on the image.
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          if face_results.detections:
            for detection in face_results.detections:
              self.mp_drawing.draw_detection(image, detection)
          if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
              self.gesture_recognizer_handler.explosion_position = self.gesture_recognizer_handler.palmPosition(hand_landmarks)
              self.mp_drawing.draw_landmarks(
                  image,
                  hand_landmarks,
                  self.mp_hands.HAND_CONNECTIONS,
                  self.mp_drawing_styles.get_default_hand_landmarks_style(),
                  self.mp_drawing_styles.get_default_hand_connections_style())
              #logic for drawing app takes index finger top point and pointing up gesture to initiate drawing
              index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
              x , y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
              if self.gesture_recognizer_handler.gesture_name.lower() == "pointing_up":
                #print("Drawing started")
                self.drawingApp.drawing = True
                self.drawingApp.draw(image, x, y)
              else:
                self.drawingApp.drawing = False
                self.drawingApp.prev_position = None
              self.drawingApp.select_color(x,y)
          self.drawingApp.draw_palette(image)
          # Flip the image horizontally for a selfie-view display.
          flipped_image = cv2.flip(image, 1)
          if flipped_image is None or flipped_image.size == 0:
            print("Error: Invalid image")
            break
          #print(f"Flipped image shape: {flipped_image.shape}")
          
          flipped_image = self.displayText(flipped_image)
          if flipped_image.shape[0] > 0 and flipped_image.shape[1] > 0:
            cv2.imshow('MediaPipe Hands', flipped_image)
          else:
            print("Error: Invalid size for display")
          if cv2.waitKey(5) & 0xFF == ord('q'):
            break
      self.cap.release()
      # Destroying All the windows 
      cv2.destroyAllWindows() 

model_path = "gesture_recognizer.task"
app = HandFaceTrackApp(model_path)
app.run()