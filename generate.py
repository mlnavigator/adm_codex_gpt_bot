import time
import openai


def set_token(token):
    openai.api_key = token


def get_messages(q, context):
    context1 = '\n'.join([d[2] for d in context])
    prompt = f"""Отвечай на вопрос пользователя только на основе информации, приведенной в контексте.
Если ответ найден, укажи из какой статьи в контексте получен ответ.
Если в контексте нет ответа на вопрос, отвечай, что у тебя нет информации по вопросу пользователя.
Контекст: {context1}
Контекст закончен.
Вопрос пользователя: {q}
Ответ: 
"""
    return [
          {
            "role": "user",
            "content": prompt
          }]


def generate(q, context, model_id='gpt-4'):
    messages = get_messages(q, context)
    for _ in range(10):
        cought = None
        try:
            res = openai.ChatCompletion.create(
                    model=model_id,
                    messages=messages, 
                    max_tokens=400,
                    temperature=0
                )
        except Exception as e:
            print(e)
            time.sleep(10 * (1 + e/3))
            cought = e
            continue
        break
        
    answer = res['choices'][0]['message']['content'] if not cought else 'К сожалению произошла техническая ошибка и ответ не был сгенерирован, обратитесь с поддержку'
    
    return answer
