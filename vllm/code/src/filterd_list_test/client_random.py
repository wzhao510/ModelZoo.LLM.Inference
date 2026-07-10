#!/usr/bin/env python3
"""Simple vLLM benchmark client — picks a random question and sends one chat request."""

import argparse
import random

from openai import OpenAI

# 10 classic questions
QUESTION_POOL = [
    # Geography & Capitals
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Brazil?",
    "What is the capital of Australia?",
    "What is the capital of Canada?",
    "What is the capital of India?",
    "What is the capital of South Korea?",
    "What is the capital of Egypt?",
    "What is the capital of Argentina?",
    "What is the capital of Nigeria?",
    # Literature & Arts
    "Who wrote Romeo and Juliet?",
    "Who wrote The Great Gatsby?",
    "Who wrote One Hundred Years of Solitude?",
    "Who painted the Mona Lisa?",
    "Who painted Starry Night?",
    "Who composed the Four Seasons?",
    "Who wrote the Iliad and the Odyssey?",
    "What is the name of the Shakesnce?",
    "Who wrote The Art of War?",
    "Who is the author of The Catcher in the Rye?",
    # Science & Astronomy
    "What is the largest planet in our solar system?",
    "How many continents are there on Earth?",
    "What is the speed of light in vacuum?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "What is the chemical symbol for silver?",
    "What is the chemical symbol for iron?",
    "What gas do plants absorb from the atmosphere during photosynthesis?",
    "What is the hardest natural substance on Earth?",
    "How many bones are in the adult human body?",
    "What is the powerhouse of the cell?",
    "What planet is known as the Red Planet?",
    # History
    "When did World War II end?",
    "When did World War I begin?",
    "Who was the first President of the United States?",
    "In what year did the Berlin Wall fall?",
    "What ancient civilization built the pyramids of Giza?",
    "Who was known as the Iron Chan",
    "What year did the Titanic sink?",
    "Which empire was ruled by Genghis Khan?",
    "What was the name of the last pharaoh of Ancient Egypt?",
    "In what year did the French Revolution begin?",
    # Geography & Nature
    "How many continents are there on Earth?",
    "What is the tallest mountain in the world?",
    "What is the longest river in the world?",
    "What is the largest ocean on Earth",
    "What is the largest desert in the world?",
    "What is the deepest ocean ",
    "Which country has the most natural lakes?",
    "Which is the smallest country in the world by area?",
    "What country has the longest coastline?",
    "What is the driest place on Earth?",
    # Technology & Inventions
    "Who invented the telephone?",
    "Who is credited with inventing the World Wide Web?",
    "What year was the first iPhone released?",
    "Who co-founded Apple Inc. ",
    "What programming language was created by Guido van Rossum?",
    "What does CPU stand for?",
    "What does RAM stand for?",
    "What does HTTP stand for?",
    "What company developed the Android operating system?",
    "In what year was the first email sent?",
    # Mathematics
    "What is the value of pi to two decimal places?",
    "What is the square root of 144?",
    "How many degrees are in a right angle?",
    "What is the sum of the interior angles of a triangle?",
    "What is the Fibonacci sequence",
    "What is the binary representation of the decimal number 10?",
    "How many sides does a hexagon",
    "What is 7 multiplied by 8?",
    "What is the only even prime number?",
    "How many zeros are in one million?",
    # Biology
    "What is the largest organ in t",
    "How many chromosomes do humans have?",
    "What is the primary function of red blood cells?",
    "What type of animal is a dolph",
    "What is the fastest land animal?",
    "What is the largest animal on Earth?",
    "How many hearts does an octopu",
    "What is the gestation period of an elephant in months?",
    "What bird is known for its ability to mimic human speech?",
    "What is the lifespan of a hone",
    # General Knowledge
    "What is the currency of the European Union?",
    "What is the most widely spoken language in the world by native speakers?",
    "How many players are on a stan?",
    "What color are the five Olympic rings?",
    "What is the boiling point of water in Celsius at sea level?",
    "What year was the United Nations founded?",
    "How many time zones are there in the world?",
    "What is the atomic number of carbon?",
    "How many strings does a standard violin have?",
    "What is the national flower of Japan?",
    # Sports & Games
    "In which sport is the term 'love' used to mean zero?",
    "How many squares are on a standard chessboard?",
    "What country hosted the first modern Olympic Games in 1896?",
    "What sport is played at Wimbledon?",
    "How many rings are on the Olympic flag?",
    "What does NBA stand for?",
    "What martial art originated in Brazil from Japanese influences?",
    "What is the maximum possible score in a single frame of bowling?",
    "What country has won the most FIFA World Cup titles?",
    "In darts, what is the highest possible score from three darts?",
]


def main():
    parser = argparse.ArgumentParser(description="vLLM benchmark client")
    parser.add_argument("--model-path", required=True, help="Model path/name for the request")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    base_url = "http://localhost:{}/v1".format(args.port)

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    question = random.choice(QUESTION_POOL)

    response = client.chat.completions.create(
        model=args.model_path,
        messages=[{"role": "user", "content": question}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        n=1,
        seed=42,
    )

    print()
    print("*" * 50)
    print("Question:", question)
    print("Answer:", response.choices[0].message.content)
    print("*" * 50)


if __name__ == "__main__":
    main()