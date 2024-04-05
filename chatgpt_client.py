import openai
from googletrans import Translator


class ChatGPTClient:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.translator = Translator()

    def query(
        self,
        question: str,
    ) -> str:

        translated_question = self.translator.translate(question).text

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model here
            messages=[
                {
                    "role": "system",
                    "content": "The following questions will be related to agriculture and sometimes it might go beyond that. Please respond the following questions with this in mind.",
                },
                {"role": "user", "content": translated_question},
            ],
        )

        response = response["choices"][0]["message"]["content"]
        
        amharic_response = self.translator.translate(response, dest="am").text

        return amharic_response
