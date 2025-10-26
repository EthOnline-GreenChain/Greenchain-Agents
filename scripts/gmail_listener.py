
#!/usr/bin/env python3
from __future__ import annotations

import argparse, base64, json, os, time, hashlib
from typing import Dict, Any, List

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from greenchain.schema import Envelope, SourceInfo, envelope_json_schema_for_asi, BillingStatement, IntervalReading
from greenchain.asi_client import ASIClient
from greenchain.pdf_text import extract_text_from_pdf

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def get_service(credentials_path: str, token_path: str):
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def list_messages_from_sender(service, sender: str, newer_than_days: int = 30, label: str | None = None):
    query = f'from:{sender} has:attachment newer_than:{newer_than_days}d'
    if label:
        query += f' label:{label}'
    results = service.users().messages().list(userId="me", q=query, maxResults=25).execute()
    return results.get("messages", [])

def get_attachments(service, msg_id: str) -> List[Dict[str, Any]]:
    msg = service.users().messages().get(userId="me", id=msg_id).execute()
    payload = msg.get("payload", {})
    parts = payload.get("parts", []) or []
    atts = []
    for p in parts:
        filename = p.get("filename")
        body = p.get("body", {})
        att_id = body.get("attachmentId")
        mime = p.get("mimeType")
        if filename and att_id:
            att = service.users().messages().attachments().get(userId="me", messageId=msg_id, id=att_id).execute()
            data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))
            atts.append({"filename": filename, "bytes": data, "mimeType": mime, "message_id": msg_id, "attachment_id": att_id})
    return atts

def process_pdf_bytes(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    text = extract_text_from_pdf(pdf_bytes)
    client = ASIClient()
    schema = envelope_json_schema_for_asi()
    prompt = (
        "You are converting utility electricity bills (PDF text) into a structured JSON payload "
        "for Greenchain. Extract billing statement fields and any interval usage present. Use ISO dates, UTC datetimes, kWh.\n\n"
        "PDF TEXT START\n"
        f"{text}\n"
        "PDF TEXT END\n"
    )
    out = client.chat_structured(prompt, schema)
    return out

def main():
    ap = argparse.ArgumentParser(description="Poll Gmail for new emails from a sender and ingest PDF attachments")
    ap.add_argument("--sender", required=True, help="Email address to watch (e.g., billing@utility.com)")
    ap.add_argument("--credentials", default="credentials.json", help="OAuth client credentials.json")
    ap.add_argument("--token", default="token.json", help="Saved token.json (created after first run)")
    ap.add_argument("--label", default=None, help="Optional Gmail label to filter")
    ap.add_argument("--newer-than-days", type=int, default=30, help="Lookback window")
    ap.add_argument("--output-dir", default="out_gmail", help="Directory for JSON outputs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    service = get_service(args.credentials, args.token)

    messages = list_messages_from_sender(service, args.sender, newer_than_days=args.newer_than_days, label=args.label)
    if not messages:
        print(json.dumps({"found": 0, "outputs": []}))
        return

    outputs = []
    for m in messages:
        msg_id = m["id"]
        atts = get_attachments(service, msg_id)
        for att in atts:
            if (att.get("mimeType") or "").lower() != "application/pdf" and not att["filename"].lower().endswith(".pdf"):
                continue
            extracted = process_pdf_bytes(att["bytes"], att["filename"])
            # Seal into Envelope
            src = SourceInfo(source_type="gmail", filename=att["filename"], sha256=sha256_bytes(att["bytes"]), extra={"message_id": msg_id, "attachment_id": att["attachment_id"]})
            env = Envelope(source=src)
            if "billing_statements" in extracted:
                env.billing_statements = [BillingStatement(**bs) for bs in extracted["billing_statements"]]
            if "interval_readings" in extracted:
                env.interval_readings = [IntervalReading(**ir) for ir in extracted["interval_readings"]]
            env.close()
            out_obj = env.model_dump()
            out_path = os.path.join(args.output_dir, f"{msg_id}_{att['filename']}.json")
            with open(out_path, "w") as w:
                json.dump(out_obj, w, indent=2, default=str)
            outputs.append(out_path)

    print(json.dumps({"found": len(outputs), "outputs": outputs}, indent=2))

if __name__ == "__main__":
    main()
