from datetime import datetime
import os
from slack_sdk import WebClient


def main() -> None:
    token = os.environ["SLACK_BOT_TOKEN"]
    channel = os.environ["SLACK_CHANNEL"]
    client = WebClient(token=token)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    repo = os.environ.get("GITHUB_REPOSITORY", "local-run")
    run_id = os.environ.get("GITHUB_RUN_ID", "n/a")
    sha = os.environ.get("GITHUB_SHA", "n/a")

    lines = [
        "Slack smoke test ✅",
        f"time: {now}",
        f"repo: {repo}",
        f"run: {run_id}",
        f"sha: {sha[:8] if sha and len(sha) >= 8 else sha}",
    ]

    message = "\n".join(lines)
    resp = client.chat_postMessage(channel=channel, text=message)
    ok = resp.get("ok", False)
    channel_id = resp.get("channel", "")
    ts = resp.get("ts", "")
    print(f"ok={ok} channel={channel_id} ts={ts}")


if __name__ == "__main__":
    main()


