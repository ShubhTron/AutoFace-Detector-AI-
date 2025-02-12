import cv2
from mtcnn import MTCNN

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    detector = MTCNN()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(image_rgb)
        
        for f in faces:
            x, y, w, h = f['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Write frame to output video
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to", output_path)

# Example usage
input_video_path = "test2.mp4" 
output_video_path = "output3_video.mp4"  # Output video file name
process_video(input_video_path, output_video_path)
