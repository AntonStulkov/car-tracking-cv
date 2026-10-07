import cv2

def run_pipline(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Ошибка: Не удалось открыть видео по пути {video_path} ')
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Видеопоток завершён')
            break

        cv2.imshow('Car Tracking Pipeline', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
  
  run_pipline('data/car_video.mp4')