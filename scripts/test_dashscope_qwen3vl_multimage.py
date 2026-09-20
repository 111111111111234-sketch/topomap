"""Low-cost direct-DashScope smoke test for Qwen3-VL multi-image input.

It sends five tiny valid PNGs, so it verifies image-count compatibility before a
long GOAT evaluation without uploading scene observations or exposing a key.
"""

import argparse
import base64
from io import BytesIO
import os

from openai import OpenAI
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument(
        "--model", default="qwen3-vl-30b-a3b-instruct", help="DashScope model ID"
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    image_buffer = BytesIO()
    Image.new("RGB", (64, 64), color=(127, 127, 127)).save(image_buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")
    content = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for _ in range(args.images)
    ]
    content.append(
        {
            "type": "text",
            "text": "Reply with exactly: OK",
        }
    )
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=16,
    )
    print(f"model={args.model}; images={args.images}; response={response.choices[0].message.content!r}")


if __name__ == "__main__":
    main()
