#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Basic example for a bot that works with polls. Only 3 people are allowed to interact with each
poll/quiz the bot generates. The preview command generates a closed poll/quiz, exactly like the
one the user sends the bot
"""
import logging
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re

from telegram import (
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

formatEx = """
1: Футболист назвал произошедшее дебильной шуткой, которая стала достоянием общественности из-за ошибки видеооператоров клубного канала сине-бело-голубых.
2: Ну, а с самыми резкими высказываниями выступил спортивный комментатор Дмитрий Губерниев, склонность которого обсуждать ДТП с участием публичных людей, не стесняясь в выражениях, стала достоянием общественности еще в 2011 году, когда в аварии погибла жена вратаря «Зенита» Вячеслава Малафеева Марина.
Слово: достояние
"""

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
tokenizer = BertTokenizer.from_pretrained("./model_save/", do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
    "./model_save/",
    num_labels=2,                       
    output_attentions=False,
    output_hidden_states=False,
)

device="cpu"
model.to(device)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inform user about what this bot can do"""
    await update.message.reply_text(
        """Добрый день!\n Я помогу вам решить задачи типа WIC в NLP. Для этого мне необходимо получить от вас информацию в формате: """ + formatEx + """Результатом будет мой ответ. Контекст един / Контекст разный """
    )

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    responce_text = "Hello"
    inputstring = return_sentence(update.message.text)
    if inputstring == "":
        responce_text = """Необходимо соблюдать формат""" + formatEx
    else:
        input, attention_mask = tokenize_sentance(inputstring)
        responce_text = "Контекст един" if inference(input, attention_mask) == 1 else "Контекст разный"
             
    await update.message.reply_text(responce_text)
    
def tokenize_sentance(sentence):
    encoded_dict = tokenizer.encode_plus(
                    sentence,
                    add_special_tokens = True,
                    max_length = 180,
                    return_attention_mask = True,
                    padding=True,
                    truncation=True,
                    return_tensors = 'pt',
            )                
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

def inference(input,attention_mask):
    b_input_ids, b_input_mask = input.to(device),attention_mask.to(device)
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logit = outputs[0].cpu().detach().numpy()[0]
    result = np.argmax(logit)
    return result

def return_sentence(msg) -> str:
    snt1, snt2, word = ("",) * 3
    lines = msg.split('\n')
    for line in lines:
        if line.startswith("1:"):
            snt1 = line.split("1:")[1].strip()
        elif line.startswith("2:"):
            snt2 = line.split("2:")[1].strip()
        elif line.startswith("Слово: "):
            word = line.split("Слово:")[1].strip()

    if snt1 == "" or snt2 == "" or word == "":
        return ""
    return f"{snt1}. {snt2}. {word}"

def main() -> None:
    """Run bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("2121440318:AAF-2SuxzIrJzVwjh-SPRhNOjRUieBw2q7w").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()