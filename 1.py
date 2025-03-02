from telethon import TelegramClient, events
from telethon.tl.types import PeerChat, PeerChannel, InputPeerChannel, InputPeerUser, InputPeerChat
import asyncio
import random
import logging
import json
import os
import time
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    filename='telegram_bot.log')
logger = logging.getLogger(__name__)

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Конфигурация бота
class BotConfig:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.load_config()
        
    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.api_id = config.get('api_id')
                self.api_hash = config.get('api_hash')
                self.session_name = config.get('session_name', 'user_session')
                self.chat_ids = config.get('chat_ids', [])
                self.response_delay = config.get('response_delay', {'min': 1, 'max': 5})
                self.message_probability = config.get('message_probability', 0.7)
                self.learning_enabled = config.get('learning_enabled', True)
                print(f"Загружена конфигурация: {self.chat_ids}")
        else:
            # Значения по умолчанию если конфигурационный файл отсутствует
            self.api_id = None  # Нужно заполнить
            self.api_hash = None  # Нужно заполнить
            self.session_name = 'user_session'
            self.chat_ids = []  # ID чатов, в которых будет работать бот
            self.response_delay = {'min': 1, 'max': 5}  # Задержка в секундах перед ответом
            self.message_probability = 0.7  # Вероятность ответа на сообщение
            self.learning_enabled = True
            self.save_config()
    
    def save_config(self):
        config = {
            'api_id': self.api_id,
            'api_hash': self.api_hash,
            'session_name': self.session_name,
            'chat_ids': self.chat_ids,
            'response_delay': self.response_delay,
            'message_probability': self.message_probability,
            'learning_enabled': self.learning_enabled
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

# Класс для хранения и обучения на сообщениях
class MessageLearner:
    def __init__(self, data_file='message_data.json'):
        self.data_file = data_file
        self.greetings = [
            "ку", "ку бро", "привет", "хай", "здарова", "йоу", "хеллоу", "салют", 
            "здравствуйте", "приветик", "дороу", "хола", "приветствую", "здрасьте"
        ]
        self.phrases = [
            "норм", "как сам", "че каво", "как дела", "что нового", "что делаешь",
            "понятно", "ясно", "согласен", "точно", "реально", "зачет", "круто",
            "да ладно", "серьезно", "жесть", "капец", "ну и ну", "офигеть", "ого"
        ]
        self.responses = defaultdict(list)
        self.word_associations = defaultdict(Counter)
        self.message_patterns = []
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.responses = defaultdict(list, data.get('responses', {}))
                    
                    # Преобразование счетчиков из словарей
                    self.word_associations = defaultdict(Counter)
                    for word, counts in data.get('word_associations', {}).items():
                        self.word_associations[word] = Counter(counts)
                    
                    self.message_patterns = data.get('message_patterns', [])
                    
                    # Добавление пользовательских приветствий, если они есть
                    custom_greetings = data.get('greetings', [])
                    if custom_greetings:
                        self.greetings.extend(custom_greetings)
                        # Уникальные значения
                        self.greetings = list(set(self.greetings))

                    # Если данных мало, добавляем базовые фразы в паттерны
                    if len(self.message_patterns) < 10:
                        self.message_patterns.extend(self.phrases)
                        
                logger.info(f"Данные успешно загружены из {self.data_file}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных: {e}")
                # Если ошибка, добавляем базовые фразы
                self.message_patterns.extend(self.phrases)
        else:
            # Если файла нет, добавляем базовые фразы
            self.message_patterns.extend(self.phrases)
    
    def save_data(self):
        data = {
            'responses': dict(self.responses),
            'word_associations': {word: dict(counter) for word, counter in self.word_associations.items()},
            'message_patterns': self.message_patterns,
            'greetings': self.greetings
        }
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Данные успешно сохранены в {self.data_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных: {e}")
    
    def learn_from_message(self, message):
        if not message or len(message) < 2:
            return
        
        # Нормализация сообщения
        message = message.lower().strip()
        
        # Токенизация сообщения
        try:
            tokens = word_tokenize(message, language='russian')
        except:
            tokens = message.split()
        
        # Добавление паттерна сообщения
        if len(message) > 3 and message not in self.message_patterns:
            self.message_patterns.append(message)
            # Ограничение количества хранимых паттернов
            if len(self.message_patterns) > 5000:
                self.message_patterns = self.message_patterns[-5000:]
        
        # Обновление ассоциаций слов
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]
            self.word_associations[current_word][next_word] += 1
        
        # Добавление приветствий
        for greeting in self.greetings:
            if message.startswith(greeting) and len(message) > len(greeting) + 2:
                response_part = message[len(greeting):].strip()
                if response_part not in self.responses[greeting]:
                    self.responses[greeting].append(response_part)
        
        # Определение типа сообщения и сохранение ответов
        if '?' in message:
            key = 'questions'
            if message not in self.responses[key]:
                self.responses[key].append(message)
        elif any(excl in message for excl in ['!', 'ого', 'вау', 'круто']):
            key = 'exclamations'
            if message not in self.responses[key]:
                self.responses[key].append(message)
        else:
            key = 'statements'
            if message not in self.responses[key]:
                self.responses[key].append(message)
    
    def get_greeting(self):
        return random.choice(self.greetings)
    
    def get_response_to_greeting(self, greeting):
        if greeting in self.responses and self.responses[greeting]:
            return random.choice(self.responses[greeting])
        else:
            return None

    def generate_response(self, input_message=None):
        # Если сообщение не предоставлено, выберем случайное из сохраненных паттернов
        if not input_message:
            if self.message_patterns:
                return random.choice(self.message_patterns)
            else:
                return random.choice(self.greetings)
        
        # Если входящее сообщение есть и оно является приветствием
        if input_message:
            input_message = input_message.lower().strip()
            
            # Проверка на приветствие
            for greeting in self.greetings:
                if input_message.startswith(greeting):
                    response = self.get_response_to_greeting(greeting)
                    if response:
                        return f"{greeting} {response}"
                    else:
                        return greeting
            
            # Генерация ответа на основе ассоциаций слов
            try:
                tokens = word_tokenize(input_message, language='russian')
            except:
                tokens = input_message.split()
                
            if tokens:
                # Выбираем случайное слово из сообщения, для которого у нас есть ассоциации
                viable_words = [word for word in tokens if word in self.word_associations and self.word_associations[word]]
                
                if viable_words:
                    start_word = random.choice(viable_words)
                    response_words = [start_word]
                    current_word = start_word
                    
                    # Генерируем последовательность слов на основе вероятностей
                    for _ in range(random.randint(3, 10)):
                        if current_word in self.word_associations and self.word_associations[current_word]:
                            # Выбор следующего слова на основе частотности
                            next_word = self.word_associations[current_word].most_common(5)
                            if next_word:
                                if len(next_word) > 2:
                                    words, weights = zip(*random.sample(next_word, min(len(next_word), 3)))
                                    current_word = random.choices(words, weights=weights)[0]
                                else:
                                    current_word = next_word[0][0]
                                response_words.append(current_word)
                            else:
                                break
                        else:
                            break
                    
                    return ' '.join(response_words)
        
        # Если не можем сгенерировать осмысленный ответ, используем сохраненные паттерны
        if self.responses['statements']:
            return random.choice(self.responses['statements'])
        elif self.message_patterns:
            return random.choice(self.message_patterns)
        else:
            return random.choice(["ну да", "согласен", "точно", "и не говори", "бывает"])
    
    def humanize_message(self, message):
        """Делает сообщение более похожим на сообщение от человека."""
        # Проверяем, что сообщение не пустое
        if not message:
            message = random.choice(self.greetings)
            
        # Иногда убираем знаки препинания
        if random.random() < 0.7:
            message = re.sub(r'[,.;:]', '', message)
        
        # Иногда добавляем эмодзи
        if random.random() < 0.2:
            emojis = ["👍", "😊", "🙂", "👌", "💪", "🔥", "😂", "👀", "🤔"]
            message += " " + random.choice(emojis)
        
        # Иногда делаем ошибки в словах (заменяем буквы)
        if random.random() < 0.1:
            words = message.split()
            if words:
                word_to_modify = random.choice(words)
                if len(word_to_modify) > 3:
                    pos = random.randint(1, len(word_to_modify) - 2)
                    word_to_modify = word_to_modify[:pos] + random.choice('йцукенгшщзхъфывапролджэячсмитьбю') + word_to_modify[pos+1:]
                    words[words.index(random.choice(words))] = word_to_modify
                    message = ' '.join(words)
        
        return message

# Основной класс бота
class HumanLikeBot:
    def __init__(self):
        self.config = BotConfig()
        self.learner = MessageLearner()
        self.client = None
        self.active_chats = {}  # Для отслеживания активных диалогов
        self.last_message_time = {}  # Для отслеживания времени последнего сообщения в чате
        self.initialized = False
    
    async def initialize(self):
        if not self.config.api_id or not self.config.api_hash:
            logger.error("API ID или API Hash не настроены. Пожалуйста, заполните config.json")
            return False
        
        # Создание клиента Telegram
        self.client = TelegramClient(self.config.session_name, self.config.api_id, self.config.api_hash)
        await self.client.start()
        
        # Проверяем доступ к чатам
        print(f"Проверка доступа к чатам: {self.config.chat_ids}")
        for chat_id in self.config.chat_ids:
            try:
                # Преобразуем ID чата в целое число
                numeric_chat_id = int(chat_id)
                
                # Получаем информацию о чате
                chat = await self.client.get_entity(numeric_chat_id)
                print(f"Успешно получен доступ к чату: {chat.id} - {getattr(chat, 'title', 'Личный чат')}")
                
                # Инициализируем время последнего сообщения
                self.last_message_time[str(chat.id)] = time.time()
            except Exception as e:
                print(f"Ошибка при доступе к чату {chat_id}: {e}")
                logger.error(f"Ошибка при доступе к чату {chat_id}: {e}")
        
        self.initialized = True
        logger.info("Бот успешно инициализирован и подключен к Telegram")
        return True
    
    def is_bot_chat(self, chat_id):
        """Проверяет, находится ли чат в списке отслеживаемых чатов."""
        return str(chat_id) in [str(cid) for cid in self.config.chat_ids]
    
    async def add_chat(self, chat_id):
        """Добавляет чат в список отслеживаемых."""
        chat_id_str = str(chat_id)
        if chat_id_str not in [str(cid) for cid in self.config.chat_ids]:
            self.config.chat_ids.append(chat_id_str)
            self.config.save_config()
            logger.info(f"Чат {chat_id} добавлен в список отслеживаемых")
            return True
        return False
    
    async def remove_chat(self, chat_id):
        """Удаляет чат из списка отслеживаемых."""
        chat_id_str = str(chat_id)
        if chat_id_str in [str(cid) for cid in self.config.chat_ids]:
            self.config.chat_ids.remove(chat_id_str)
            self.config.save_config()
            logger.info(f"Чат {chat_id} удален из списка отслеживаемых")
            return True
        return False
    
    async def process_incoming_message(self, event):
        """Обрабатывает входящие сообщения."""
        try:
            # Получаем информацию о чате
            chat = await event.get_chat()
            chat_id = str(getattr(chat, 'id', None))
            
            # Выводим информацию о полученном сообщении
            sender = await event.get_sender()
            sender_id = getattr(sender, 'id', 'Unknown')
            is_self = sender and sender.is_self
            message_text = event.message.message
            print(f"Сообщение от {sender_id} в чате {chat_id}: {message_text}")
            print(f"Это собственное сообщение: {is_self}")
            print(f"Чат в списке: {self.is_bot_chat(chat_id)}")
            
            # Проверяем, является ли это сообщение из отслеживаемого чата
            if not self.is_bot_chat(chat_id):
                print(f"Чат {chat_id} не в списке отслеживаемых")
                return
            
            # Не реагируем на собственные сообщения
            if is_self:
                print("Пропускаем собственное сообщение")
                return
            
            # Обучаемся на входящем сообщении, если обучение включено
            if self.config.learning_enabled and message_text:
                self.learner.learn_from_message(message_text)
                # Сохраняем данные каждые N сообщений
                if random.random() < 0.1:  # ~10% шанс сохранения после каждого сообщения
                    self.learner.save_data()
            
            # Обновляем время последнего сообщения в чате
            self.last_message_time[chat_id] = time.time()
            
            # Решаем, отвечать ли на сообщение
            should_respond = random.random() < self.config.message_probability
            
            if should_respond:
                print(f"Будем отвечать на сообщение в чате {chat_id}")
                # Добавляем случайную задержку перед ответом для имитации человека
                delay = random.uniform(
                    self.config.response_delay['min'],
                    self.config.response_delay['max']
                )
                
                # Иногда отображаем "печатает..." статус
                try:
                    if random.random() < 0.8:
                        async with self.client.action(chat, 'typing'):
                            await asyncio.sleep(delay)
                            
                            # Генерируем ответ
                            response = self.learner.generate_response(message_text)
                            
                            # Делаем ответ более "человечным"
                            response = self.learner.humanize_message(response)
                            
                            # Отправляем ответ
                            await self.client.send_message(chat, response)
                            print(f"Отправлен ответ в чат {chat_id}: {response}")
                            logger.info(f"Отправлен ответ в чат {chat_id}: {response}")
                    else:
                        # Просто ждем без статуса печатания
                        await asyncio.sleep(delay)
                        
                        # Генерируем ответ
                        response = self.learner.generate_response(message_text)
                        
                        # Делаем ответ более "человечным"
                        response = self.learner.humanize_message(response)
                        
                        # Отправляем ответ
                        await self.client.send_message(chat, response)
                        print(f"Отправлен ответ в чат {chat_id}: {response}")
                        logger.info(f"Отправлен ответ в чат {chat_id}: {response}")
                except Exception as e:
                    print(f"Ошибка при отправке ответа в чат {chat_id}: {e}")
                    logger.error(f"Ошибка при отправке ответа в чат {chat_id}: {e}")
        except Exception as e:
            print(f"Ошибка при обработке сообщения: {e}")
            logger.error(f"Ошибка при обработке сообщения: {e}")
    
    async def initiate_conversation(self):
        """Иногда самостоятельно начинает диалог в чатах."""
        print("Запущен инициатор диалога")
        while True:
            # Проверяем каждый чат
            for chat_id in self.config.chat_ids:
                current_time = time.time()
                last_time = self.last_message_time.get(chat_id, 0)
                
                # Если прошло достаточно времени с последнего сообщения
                if current_time - last_time > random.randint(600, 3600):  # 10 минут - 1 час
                    # С некоторой вероятностью инициируем разговор
                    if random.random() < 0.3:  # 30% шанс
                        try:
                            # Генерируем сообщение
                            if random.random() < 0.5:
                                message = self.learner.get_greeting()
                            else:
                                message = self.learner.generate_response()
                            
                            # Делаем сообщение более "человечным"
                            message = self.learner.humanize_message(message)
                            
                            # Получаем сущность чата
                            try:
                                entity = await self.client.get_entity(int(chat_id))
                                print(f"Получена сущность чата {chat_id}: {entity}")
                            except Exception as e:
                                print(f"Ошибка при получении сущности чата {chat_id}: {e}")
                                continue
                            
                            # Иногда отображаем "печатает..." статус
                            if random.random() < 0.8:
                                async with self.client.action(entity, 'typing'):
                                    await asyncio.sleep(random.uniform(1, 3))
                                    await self.client.send_message(entity, message)
                                    print(f"Инициировано сообщение в чате {chat_id}: {message}")
                            else:
                                await self.client.send_message(entity, message)
                                print(f"Инициировано сообщение в чате {chat_id}: {message}")
                            
                            # Обновляем время последнего сообщения
                            self.last_message_time[chat_id] = current_time
                            logger.info(f"Инициирован разговор в чате {chat_id}")
                        except Exception as e:
                            print(f"Ошибка при инициировании разговора в чате {chat_id}: {e}")
                            logger.error(f"Ошибка при инициировании разговора в чате {chat_id}: {e}")
            
            # Ждем случайное время перед следующей проверкой
            await asyncio.sleep(random.randint(300, 900))  # 5-15 минут
    
    async def run(self):
        """Запускает бота."""
        # Инициализация
        initialized = await self.initialize()
        if not initialized:
            print("Ошибка инициализации бота")
            return
        
        # Регистрируем обработчик входящих сообщений
        print("Регистрация обработчика сообщений")
        
        @self.client.on(events.NewMessage)
        async def message_handler(event):
            try:
                await self.process_incoming_message(event)
            except Exception as e:
                print(f"Ошибка при обработке сообщения: {e}")
                logger.error(f"Ошибка при обработке сообщения: {e}")
        
        # Запускаем инициатор разговора в отдельной задаче
        print("Запуск инициатора диалога")
        asyncio.create_task(self.initiate_conversation())
        
        # Отправляем приветственное сообщение в каждый чат при запуске
        print("Отправка приветственных сообщений")
        for chat_id in self.config.chat_ids:
            try:
                entity = await self.client.get_entity(int(chat_id))
                greeting = self.learner.humanize_message(self.learner.get_greeting())
                await self.client.send_message(entity, greeting)
                print(f"Отправлено приветствие в чат {chat_id}: {greeting}")
                logger.info(f"Отправлено приветствие в чат {chat_id}: {greeting}")
            except Exception as e:
                print(f"Ошибка при отправке приветствия в чат {chat_id}: {e}")
                logger.error(f"Ошибка при отправке приветствия в чат {chat_id}: {e}")
        
        print("Бот запущен и готов к работе")
        logger.info("Бот запущен и готов к работе")
        
        # Поддерживаем работу бота до закрытия
        await self.client.run_until_disconnected()

# Функция для запуска бота
async def main():
    bot = HumanLikeBot()
    await bot.run()

# Вспомогательная функция для настройки конфигурации
def setup_config():
    config = BotConfig()
    
    print("Настройка бота Telegram")
    print("=======================")
    
    api_id = input("Введите API ID: ")
    api_hash = input("Введите API Hash: ")
    session_name = input("Введите имя сессии (по умолчанию 'user_session'): ") or 'user_session'
    
    print("\nНастройка чатов")
    print("Введите ID чатов, в которых будет работать бот (по одному, для завершения введите пустую строку)")
    
    chat_ids = []
    while True:
        chat_id = input("ID чата: ")
        if not chat_id:
            break
        chat_ids.append(chat_id)
    
    print("\nНастройка поведения")
    min_delay = float(input("Минимальная задержка перед ответом (сек, по умолчанию 1): ") or 1)
    max_delay = float(input("Максимальная задержка перед ответом (сек, по умолчанию 5): ") or 5)
    prob = float(input("Вероятность ответа на сообщение (0-1, по умолчанию 0.7): ") or 0.7)
    
    learning = input("Включить обучение (y/n, по умолчанию y): ").lower() != 'n'
    
    # Сохраняем настройки
    config.api_id = api_id
    config.api_hash = api_hash
    config.session_name = session_name
    config.chat_ids = chat_ids
    config.response_delay = {'min': min_delay, 'max': max_delay}
    config.message_probability = prob
    config.learning_enabled = learning
    config.save_config()
    
    print("\nНастройка завершена. Конфигурация сохранена в config.json")

# Точка входа для запуска бота
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        setup_config()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Бот остановлен")