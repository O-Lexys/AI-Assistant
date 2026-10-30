"""Приклад плагіна. Щоб увімкнути: скопіюй у plugins/ під іншою назвою (без «_» на початку), напр. plugins/hello.py,
і перезапусти асистента. Файли, що починаються з «_», не завантажуються.

Плагін — звичайний Python-файл із функцією setup(api). Усе нижче необов'язкове: бери лише потрібне.
Налаштування плагіна — у config.toml:   [plugins.hello]  name = "Саша"      (api.cfg["name"])
Вимкнути, не видаляючи:                 [plugins]  disabled = ["hello"]
"""
import datetime


def setup(api):
    # 1. Інструмент для LLM: вона сама вирішує, коли його викликати. Слова-тригери вмикають інструменти для цього запиту.
    @api.tool("coin_flip", "Flip a coin and return heads or tails.", triggers=["монет", "орел", "решк"], followup=True)
    def coin_flip(args):
        import random
        return random.choice(["орел", "решка"])

    # 2. Голосова команда без LLM (миттєво). m — результат re.match по тексту малими літерами.
    #    light=True — можна виконувати прямо в легкому слухачі, без запуску whisper.
    @api.command(r"(який|скажи) (зараз )?день тижня", light=True)
    def weekday(m, text, speaker):
        days = ["понеділок", "вівторок", "середа", "четвер", "пʼятниця", "субота", "неділя"]
        speaker.say(f"Сьогодні {days[datetime.date.today().weekday()]}.")

    # 3. Реакція на події: wake (почуто слово), heard (текст команди), reply (відповідь), ui (кожна подія інтерфейсу).
    @api.on("heard")
    def log_heard(text):
        (api.state_dir).mkdir(parents=True, exist_ok=True)
        with open(api.state_dir / "heard.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now():%F %T}  {text}\n")

    # 4. Подія інтерфейсу з плагіна (бульбашка, картка, ефект …): api.ui(t="bubble", text="Привіт", add=True)
