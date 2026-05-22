import cv2
import os
import sys
import time

DATASET_DIR = "dataset"
DET_MODEL = "models/face_detection_yunet.onnx"

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # Запрашиваем имя нового пользователя
    print("┌────────────────────────────────────────────────────────┐")
    print("│         РЕГИСТРАЦИЯ НОВОГО ПОЛЬЗОВАТЕЛЯ СКУД           │")
    print("└────────────────────────────────────────────────────────┘")
    name = input(" Enter user name (на латинице, например: sultan): ").strip()
    
    if not name:
        print("[ОШИБКА] Имя не может быть пустым!")
        return
        
    filename = os.path.join(DATASET_DIR, f"{name}.jpg")
    
    # Инициализируем детектор лиц YuNet (на разрешение 320x240)
    detector = cv2.FaceDetectorYN.create(DET_MODEL, "", (320, 240), 0.7, 0.3, 50)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("[ОШИБКА] Не удалось запустить веб-камеру!")
        return

    print(f"\n Камера активна. Смотрите прямо в объектив.")
    print(" Скрипт автоматически поймает идеальный кадр...")
    
    countdown = 3
    last_countdown_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ОШИБКА] Не удалось получить кадр с камеры.")
                break

            # Детекция лица на текущем кадре
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)

            # Если лицо обнаружено в кадре
            if faces is not None and len(faces) > 0:
                # Рисуем уведомление прямо в консоли (или на экране, если есть GUI)
                current_time = time.time()
                
                if current_time - last_countdown_time >= 1.0:
                    print(f" [ФИКСАЦИЯ] Лицо найдено! Запись через {countdown}...")
                    countdown -= 1
                    last_countdown_time = current_time
                
                if countdown <= 0:
                    # Сохраняем чистый оригинальный кадр без рамок
                    cv2.imwrite(filename, frame)
                    print("\n┌────────────────────────────────────────────────────────┐")
                    print("│ 🎉 [УСПЕХ] БИОМЕТРИЧЕСКИЙ ШАБЛОН СОХРАНЕН!             │")
                    print(f"│ Путь к файлу: {filename:<41} │")
                    print("└────────────────────────────────────────────────────────┘")
                    break
            else:
                countdown = 3  # Сбрасываем таймер, если человек отвернулся
                print(" [ПОИСК] Наведите камеру на лицо... Кадр пуст. ", end="\r")
                
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n [ОТМЕНА] Регистрация прервана пользователем.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
