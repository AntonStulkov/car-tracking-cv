# Клонирование репозитория
git clone https://github.com/AntonStulkov/car-tracking-cv.git

# Переход в папку проекта
cd car-tracking-cv

# Создание виртуального окружения
python3 -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск проекта (предварительно убедись, что видеофайл находится в папка data/car_video.mp4)
python3 main.py
