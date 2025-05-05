import json
import logging
import os
import re
import sys

import requests
import structlog

CACHE_DIR = os.environ["CACHE_DIR"]
MODEL_ID = "Qwen2.5-7B-Instruct"
MODEL_NAME = MODEL_ID.split("/")[1]

# Update the home dir before importing the huggingface lib
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, MODEL_NAME)  # noqa type

import transformers  # noqa

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.DEBUG,
)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
transformers.logging.set_verbosity_debug()
logger = structlog.get_logger(module=__name__)


def main():
    profile_path = os.getenv("PROFILE_PATH")
    config_path = os.getenv("CONFIG_PATH")
    notify_url = os.getenv("NOTIFIER_URL")

    with open(profile_path, "r") as file_obj:
        profile = json.load(file_obj)

    with open(config_path, "r") as file_obj:
        config = json.load(file_obj)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=config["accessToken"]
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, token=config["accessToken"]
    )
    pipeline = transformers.pipeline("text-generation",
                                     model=model,
                                     tokenizer=tokenizer,
                                     device_map="auto")

    messages = [
        {"role": "system", "content": config["llmSystemPrompt"]},
        {"role": "user", "content": profile["text"]},
    ]

    logger.debug("Running LLM inference",
                 system_prompt=config["llmSystemPrompt"],
                 positive_regex=config["llmPositiveResponseRegex"],
                 user_prompt=profile["text"])

    outputs = pipeline(messages, max_new_tokens=8)

    logger.debug("Got LLM outputs", outputs=outputs)

    positive_pattern = re.compile(config["llmPositiveResponseRegex"])
    output = outputs[0]["generated_text"][-1]

    if re.fullmatch(positive_pattern, output):
        logger.info("Profile ACCEPTED by assessor", **profile["metadata"])
        resp = requests.post(notify_url, json=profile)
        resp.raise_for_status()
    else:
        logger.info("Profile REJECTED by assessor", **profile["metadata"])


if __name__ == "__main__":
    main()
