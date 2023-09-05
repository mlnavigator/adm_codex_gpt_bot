import asyncio
import logging
import json
import os
import time

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message

# from typing import Callable, Dict, Any, Awaitable
from aiogram.types import TelegramObject

from concurrent.futures import ThreadPoolExecutor

from get_context import get_context
from generate import generate, set_token


base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, 'config.json'), 'r') as f:
    json_config = json.load(f)

TOKEN = json_config['tg_bot_token']
OPENAI_TOKEN = json_config['open_ai_token']

set_token(OPENAI_TOKEN)

t_pool = ThreadPoolExecutor(5)

router = Router()

    
@router.message(Command(commands=["start"]))
async def command_start_handler(message: Message, **kwargs) -> None:
    """
    This handler receive messages with `/start` command
    """
    
    welcome_text =f"""Я бот, отвечающий на вопросы по КОАП. Напишите ваш вопрос и я постараюсь найти ответ в КОАП.
    """

    await message.answer(welcome_text)
    

@router.message()
async def reply_message(message: Message, **kwargs) -> None:
    # print(kwargs['user'])
    await message.answer('Генерирую ответ')
    q = message.text
    context = get_context(q)
    f = t_pool.submit(generate, q=q, context=context)
    answ = f.result()
    await message.answer(answ)


async def main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()
    
    # ... and all other routers should be attached to Dispatcher
    dp.include_router(router)

    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(TOKEN, parse_mode="HTML")
    # And the run events dispatching
    await dp.start_polling(bot)
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            print(e)
            time.sleep(10)
