for running bot execute:
python3 tg_bot.py

For updating data 
    - download kodex and put text into assets/kodap.txt
    - run python3 prepare_data.py

Bot uses vectorizations:
    - tf-idf
    - bert LABSE from huggingface hub
    
Bot uses gpt-4 OpenAI model for generating anwer uppon extracted context.
You need OpenAI api token.

Bot doesn't use Llamaindex or other frameworks.
