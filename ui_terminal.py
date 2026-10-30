#!/usr/bin/env python3
"""Приклад ВЛАСНОГО інтерфейсу: найпростіший, у терміналі (лише стандартна бібліотека).

Підключення — config.toml:
    [ui]
    command = "python3 ~/шлях/до/examples/ui_terminal.py"

Асистент запускає цю команду замість вбудованого талісмана, а сам лише шле події у Unix-датаграмний сокет,
шлях до якого лежить у змінній ASSISTANT_UI_SOCK. Інтерфейс має створити (bind) цей сокет і читати JSON-події.
Повний перелік подій — у README.txt (розділ ПЛАГІНИ І СВІЙ ІНТЕРФЕЙС). Подія {"t": "quit"} — завершитись.
"""
import json, os, socket, sys

path = os.environ["ASSISTANT_UI_SOCK"]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(path)
print("[ui] чекаю на події…", flush=True)
while True:
    msg = json.loads(sock.recv(65536))
    if msg.get("t") == "quit":
        break
    if "s" in msg:                                    # стан: listen | think | speak | hide | idle
        print(f"[ui] стан: {msg['s']}", flush=True)
    t = msg.get("t")
    if t == "bubble":                                 # речення, яке зараз говорить асистент
        print(f"[ui] 💬 {msg['text']}", flush=True)
    elif t == "card":                                 # картка: заголовок і рядки
        print(f"[ui] ▤ {msg.get('title', '')}: " + " | ".join(msg.get("lines", [])), flush=True)
    elif t == "glyph":                                # ok | err | open | ask | bang | screen | note | timer
        print(f"[ui] знак: {msg['g']}", flush=True)
    # t == "lv" — рівні звуку (24 числа 0..1) приходять багато разів на секунду; тут їх ігноруємо
