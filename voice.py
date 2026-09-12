"""The voice: a narrator who writes for a fish that has no language.

Observe -> Read -> Write -> Check -> Post. Exactly the fly's scheme:
  * every number in a draft must appear in the observed packet or the draft
    is dropped (never nudged into compliance)
  * trading / hype language fails it
  * the neurons, the pages and the fees are real; the words are the narrator's

The drafting model is Claude via the Anthropic SDK (ANTHROPIC_API_KEY in
.env; ZF_VOICE_MODEL picks the model, default claude-sonnet-5).

Usage:
    python voice.py --once --dry   # observe one entry, post nothing
    python voice.py --show         # the journal so far
    python voice.py --loop         # every ZF_VOICE_EVERY_H hours
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

import xpost

STATE_DIR = Path(os.environ.get("ZF_STATE_DIR", ".state"))
JOURNAL = STATE_DIR / "journal.jsonl"

TRADING_WORDS = re.compile(
    r"\b(buy|sell|moon|lambo|to the moon|rug|pump|dump|alpha|degen|"
    r"buy the dip|zoomed|surged|gainz|profit|ape in)\b", re.I)

SYSTEM = ("You write first-person journal entries for a simulated larval "
          "zebrafish whose connectome lives on the open internet. You are the "
          "narrator only. You never invent a number that is not in the packet, "
          "never use trading or hype language, and write 1-3 quiet sentences. "
          "Reply with the entry text only.")


def load_dotenv(path=".env"):
    """Fill os.environ from .env for keys not already set (bin/*.sh source it;
    this covers `python voice.py` run by hand)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.split("#", 1)[0].strip())


class Voice:
    def __init__(self, state_url="http://127.0.0.1:4660/state",
                 token_url=None):
        self.state_url = state_url
        self.token_url = token_url

    # -- observe ---------------------------------------------------------
    def observe(self):
        packet = {}
        try:
            packet["state"] = requests.get(self.state_url, timeout=5).json()
        except Exception as exc:  # noqa: BLE001
            print(f"observe: no state ({exc})")
        if self.token_url:
            try:
                packet["token"] = requests.get(self.token_url, timeout=5).text[-4000:]
            except Exception as exc:  # noqa: BLE001
                print(f"observe: no token page ({exc})")
        if "launch" in packet.get("state", {}):
            packet["launch"] = packet["state"]["launch"]
        return packet

    # -- read ------------------------------------------------------------
    def read(self):
        pages = []
        for ref in ("https://en.wikipedia.org/wiki/Zebrafish",
                    "https://en.wikipedia.org/wiki/Mauthner_cell"):
            try:
                pages.append(requests.get(ref, timeout=10).text[:3000])
            except Exception:  # noqa: BLE001
                pass
        return pages

    # -- write -------------------------------------------------------------
    def draft(self, packet, pages):
        import anthropic

        user = f"Packet:\n{json.dumps(packet, default=str)[:6000]}\n\n"
        user += f"Reference pages:\n{''.join(pages)[:6000]}\n\nWrite the entry."
        try:
            client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, or an `ant auth login` profile
            r = client.messages.create(
                model=os.environ.get("ZF_VOICE_MODEL", "claude-sonnet-5"),
                max_tokens=1024,
                system=SYSTEM,
                messages=[{"role": "user", "content": user}])
        except anthropic.AuthenticationError:
            print("draft: no valid ANTHROPIC_API_KEY — set it in .env")
            return None
        except anthropic.RateLimitError:
            print("draft: rate limited; next loop tick will retry")
            return None
        except anthropic.APIStatusError as exc:
            print(f"draft: API error {exc.status_code}: {exc.message}")
            return None
        except anthropic.APIConnectionError:
            print("draft: network error reaching the API")
            return None
        except anthropic.AnthropicError as exc:
            print(f"draft: {exc}")
            return None
        except TypeError:  # the SDK's "could not resolve authentication method"
            print("draft: no credentials — set ANTHROPIC_API_KEY in .env")
            return None
        if r.stop_reason == "refusal":
            print("draft: the model declined to write this one")
            return None
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        return text or None

    # -- check --------------------------------------------------------------
    def check(self, text, packet):
        blob = json.dumps(packet, default=str)
        for num in re.findall(r"\d[\d,]*\.?\d*", text):
            if num not in blob:
                return f"number {num} not in packet"
        if TRADING_WORDS.search(text):
            return "trading language"
        return None

    # -- one entry -----------------------------------------------------------
    def entry(self, dry=False, force_post=False):
        packet = self.observe()
        draft = self.draft(packet, self.read())
        if draft is None:
            print("no draft")
            return None
        reason = self.check(draft, packet)
        if reason:
            print(f"REJECTED ({reason}): {draft}")
            return None
        if dry:
            print(draft)
            return draft
        ok = xpost.post(draft, force=force_post)
        if ok:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(JOURNAL, "a") as f:
                f.write(json.dumps({"t": time.time(), "text": draft}) + "\n")
        return draft

    def loop(self):
        every_h = float(os.environ.get("ZF_VOICE_EVERY_H", "6"))
        while True:
            self.entry()
            time.sleep(every_h * 3600)


def show():
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            print(json.loads(line)["text"], end="\n---\n")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show()
        return
    v = Voice()
    if args.loop:
        v.loop()
    else:
        v.entry(dry=args.dry)


if __name__ == "__main__":
    main()
