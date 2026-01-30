"""
Q-Nav Object Detection Module
Detects obstacles using YOLO AI
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

class DebrisDetector:
    """Detects space debris/obstacles"""
    
    def __init__(self):
        print("🚀 Starting Q-Nav Detector...")
        
        # Load YOLO AI model
        print("📥 Loading AI model (first time downloads ~6MB)...")
        self.model = YOLO('yolov8n.pt')
        
        # Check GPU/CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Using: {self.device}")
        
        # Statistics
        self.total_detections = 0
        self.total_frames = 0
        
        print("✓ Ready!\n")
    
    def detect(self, frame):
        """
        Detect objects in frame
        
        Returns:
            detections: List of objects found
            annotated: Frame with boxes drawn
        """
        # Run AI detection
        results = self.model.predict(frame, conf=0.3, verbose=False)
        
        # Extract results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().item()
                class_id = int(box.cls[0].cpu().item())
                class_name = self.model.names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
        
        # Get annotated frame
        annotated = results[0].plot()
        
        # Update stats
        self.total_detections += len(detections)
        self.total_frames += 1
        
        return detections, annotated
    
    def calculate_risk(self, detections, frame_shape):
        """
        Calculate collision risk
        
        Returns:
            risk_level: 'HIGH', 'MEDIUM', 'LOW', 'SAFE'
            risk_score: 0-100
        """
        if not detections:
            return 'SAFE', 0
        
        height, width = frame_shape[:2]
        center_x, center_y = width // 2, height // 2
        
        max_risk = 0
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Size score (bigger = more danger)
            size = ((x2 - x1) * (y2 - y1)) / (width * height)
            size_score = size * 100
            
            # Proximity score (center = more danger)
            obj_x = (x1 + x2) / 2
            obj_y = (y1 + y2) / 2
            distance = np.sqrt((obj_x - center_x)**2 + (obj_y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            proximity_score = (1 - distance / max_dist) * 50
            
            # Total risk
            risk = size_score + proximity_score
            max_risk = max(max_risk, risk)
        
        # Classify risk level
        if max_risk > 70:
            return 'HIGH', int(max_risk)
        elif max_risk > 40:
            return 'MEDIUM', int(max_risk)
        elif max_risk > 20:
            return 'LOW', int(max_risk)
        else:
            return 'SAFE', int(max_risk)
    
    def run_webcam(self):
        """Run detection on webcam"""
        print("📹 Opening webcam...")
        print("Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access webcam!")
            return
        
        print("✓ Webcam active!\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            detections, annotated = self.detect(frame)
            
            # Calculate risk
            risk_level, risk_score = self.calculate_risk(detections, frame.shape)
            
            # Color based on risk
            colors = {
                'SAFE': (0, 255, 0),
                'LOW': (0, 255, 255),
                'MEDIUM': (0, 165, 255),
                'HIGH': (0, 0, 255)
            }
            color = colors[risk_level]
            
            # Add text to frame
            cv2.putText(annotated, f"RISK: {risk_level}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(annotated, f"Score: {risk_score}/100", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated, f"Objects: {len(detections)}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Show
            cv2.imshow('Q-Nav Detection', annotated)
            
            # Print
            if detections:
                print(f"Frame {self.total_frames}: {len(detections)} objects | Risk: {risk_level} ({risk_score})")
            
            # Quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print(f"\n{'='*50}")
        print("DETECTION COMPLETE")
        print(f"{'='*50}")
        print(f"Frames processed: {self.total_frames}")
        print(f"Total detections: {self.total_detections}")
        print(f"Average per frame: {self.total_detections/max(self.total_frames,1):.2f}")
        print(f"{'='*50}\n")


# Run the detector
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Q-NAV OBJECT DETECTION")
    print("="*50 + "\n")
    
    detector = DebrisDetector()
    detector.run_webcam()
    
    print("✓ Done! 🎉\n")