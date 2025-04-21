import json
import os
import re

import requests
import structlog
import torch
import transformers

logger = structlog.getLogger(__name__)
structlog.configure(processors=[structlog.processors.JSONRenderer()])
transformers.logging.set_verbosity_debug()


def main():
    os.environ["HF_HOME"] = os.getenv("CACHE_DIR")
    profile_path = os.getenv("PROFILE_PATH")
    config_path = os.getenv("CONFIG_PATH")
    notify_url = os.getenv("NOTIFIER_URL")

    with open(profile_path, "r") as file_obj:
        profile = json.load(file_obj)

    with open(config_path, "r") as file_obj:
        config = json.load(file_obj)

    model = transformers.AutoModel.from_pretrained(
        "meta-llama/Llama-3.3-70B-Instruct",
        token=config["accessToken"],
        torch_dtype=torch.bfloat16,
    )
    pipeline = transformers.pipeline("text-generation",
                                     model=model,
                                     device_map="auto")

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
