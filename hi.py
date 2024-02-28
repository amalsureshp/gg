
import cv2
import numpy as np
import pygame

# Initialize Pygame and load sounds
pygame.init()
sounds = [pygame.mixer.Sound('sound1.wav'), pygame.mixer.Sound('sound2.wav'), pygame.mixer.Sound('sound3.wav')]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    _, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find contour of max area (hand)
    max_area = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            ci = i

    # Extract hand and find convex hull
    hand = contours[ci]
    hull = cv2.convexHull(hand)

    # Find moments
    moments = cv2.moments(hand)
    if moments['m00'] != 0:
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # Center point falls in different regions of the screen
        if center[1] < 150:
            sounds[0].play()
        elif center[1] > 250:
            sounds[1].play()
        else:
            sounds[2].play()

    # Display the resulting frame
    cv2.imshow('Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
