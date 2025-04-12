import json
import os
import re

import requests
import structlog
import torch
import transformers

logger = structlog.getLogger(__name__)
structlog.configure(processors=[structlog.processors.JSONRenderer()])


def main():
    profile_path = os.getenv("PROFILE_PATH")
    config_path = os.getenv("CONFIG_PATH")
    notify_url = os.getenv("NOTIFIER_URL")

    with open(profile_path, "r") as file_obj:
        profile = json.load(file_obj)

    with open(config_path, "r") as file_obj:
        config = json.load(file_obj)

    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Llama-3.3-70B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16,
                      "token": config["accessToken"]},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": config["systemPrompt"]},
        {"role": "user", "content": profile["text"]},
    ]

    logger.debug("Running LLM inference",
                 system_prompt=config["systemPrompt"],
                 positive_regex=config["positiveRegex"],
                 user_prompt=profile["text"])

    outputs = pipeline(messages, max_new_tokens=8)

    positive_pattern = re.compile(config["positiveRegex"])
    output = outputs[0]["generated_text"][-1]

    if re.fullmatch(positive_pattern, output):
        logger.info("Profile ACCEPTED by assessor", **profile["metadata"])
        resp = requests.post(notify_url, json=profile)
        resp.raise_for_status()
    else:
        logger.info("Profile REJECTED by assessor", **profile["metadata"])


if __name__ == "__main__":
    main()
