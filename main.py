import cv

# Load the video feed from the camera
cap = cv.VideoCapture(0)  # Use 0 for the default camera, or change to the index of your camera

# Initialize variables for object detection and tracking
object_detector = cv.createBackgroundSubtractorMOG2()  # Use a background subtractor to detect moving objects
object_count = 0  # Counter for the number of objects passing through the door

while True:
    # Read the next frame from the video feed
    ret, frame = cap.read()
    
    # Apply object detection to the frame
    mask = object_detector.apply(frame)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Iterate over each detected object
    for contour in contours:
        # Calculate the area of the contour
        area = cv.contourArea(contour)
        
        # Ignore small objects
        if area > 500:
            # Draw a bounding box around the object
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Increment the object count
            object_count += 1
    
    # Display the number of objects counted on the video feed
    cv.putText(frame, f'Object count: {object_count}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame with object bounding boxes and object count
    cv.imshow('Object detection', frame)
    
    # Check for user input to exit the loop
    if cv.waitKey(1) == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv.destroyAllWindows()
