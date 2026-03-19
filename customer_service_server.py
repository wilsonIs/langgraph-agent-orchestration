from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from advanced_qa_agent import (
    build_graph,
    resume_customer_turn,
    run_customer_turn,
    stream_customer_turn,
    stream_resume_turn,
)


LOGGER = logging.getLogger("customer_service_server")
WEB_DIR = Path(__file__).resolve().parent / "web"
GRAPH = build_graph()
GRAPH_LOCK = threading.RLock()


def json_response(
    handler: BaseHTTPRequestHandler,
    payload: dict,
    *,
    status: HTTPStatus = HTTPStatus.OK,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class CustomerServiceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CustomerServiceAgent/1.0"

    def log_message(self, format: str, *args) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            json_response(self, {"status": "ok"})
            return

        if path == "/":
            self.serve_static("index.html")
            return

        safe_path = path.lstrip("/")
        if safe_path:
            self.serve_static(safe_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            payload = self.read_json()
        except ValueError as exc:
            json_response(
                self,
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            if parsed.path == "/api/chat":
                response = self.handle_chat(payload)
                json_response(self, response)
                return

            if parsed.path == "/api/chat/stream":
                self.handle_chat_stream(payload)
                return

            if parsed.path == "/api/review":
                response = self.handle_review(payload)
                json_response(self, response)
                return

            if parsed.path == "/api/review/stream":
                self.handle_review_stream(payload)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # pragma: no cover - defensive server boundary
            LOGGER.exception("Request handling failed path=%s", parsed.path)
            if parsed.path.endswith("/stream"):
                if not getattr(self, "_stream_started", False):
                    self.start_ndjson_stream()
                self.write_stream_event({"type": "error", "error": f"服务端异常：{exc}"})
                return
            json_response(
                self,
                {"error": f"服务端异常：{exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def serve_static(self, relative_path: str) -> None:
        file_path = (WEB_DIR / relative_path).resolve()
        if WEB_DIR not in file_path.parents and file_path != WEB_DIR:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        mime_type, _ = mimetypes.guess_type(file_path.name)
        content_type = mime_type or "application/octet-stream"
        body = file_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("请求体为空。")

        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc

    def handle_chat(self, payload: dict) -> dict:
        message = str(payload.get("message", "")).strip()
        thread_id = str(payload.get("thread_id", "")).strip()
        user_id = str(payload.get("user_id", "")).strip()

        if not message:
            raise ValueError("message 不能为空。")
        if not thread_id:
            raise ValueError("thread_id 不能为空。")
        if not user_id:
            raise ValueError("user_id 不能为空。")

        with GRAPH_LOCK:
            return run_customer_turn(
                GRAPH,
                thread_id=thread_id,
                user_id=user_id,
                message=message,
            )

    def handle_chat_stream(self, payload: dict) -> None:
        message = str(payload.get("message", "")).strip()
        thread_id = str(payload.get("thread_id", "")).strip()
        user_id = str(payload.get("user_id", "")).strip()

        if not message:
            raise ValueError("message 不能为空。")
        if not thread_id:
            raise ValueError("thread_id 不能为空。")
        if not user_id:
            raise ValueError("user_id 不能为空。")

        self.start_ndjson_stream()

        def emit_event(event: dict) -> None:
            self.write_stream_event(event)

        with GRAPH_LOCK:
            stream_customer_turn(
                GRAPH,
                thread_id=thread_id,
                user_id=user_id,
                message=message,
                emit_event=emit_event,
            )

    def handle_review(self, payload: dict) -> dict:
        thread_id = str(payload.get("thread_id", "")).strip()
        user_id = str(payload.get("user_id", "")).strip()
        approved = bool(payload.get("approved"))
        reviewer_note = str(payload.get("reviewer_note", "")).strip()

        if not thread_id:
            raise ValueError("thread_id 不能为空。")
        if not user_id:
            raise ValueError("user_id 不能为空。")

        with GRAPH_LOCK:
            return resume_customer_turn(
                GRAPH,
                thread_id=thread_id,
                user_id=user_id,
                approved=approved,
                reviewer_note=reviewer_note,
            )

    def handle_review_stream(self, payload: dict) -> None:
        thread_id = str(payload.get("thread_id", "")).strip()
        user_id = str(payload.get("user_id", "")).strip()
        approved = bool(payload.get("approved"))
        reviewer_note = str(payload.get("reviewer_note", "")).strip()

        if not thread_id:
            raise ValueError("thread_id 不能为空。")
        if not user_id:
            raise ValueError("user_id 不能为空。")

        self.start_ndjson_stream()

        def emit_event(event: dict) -> None:
            self.write_stream_event(event)

        with GRAPH_LOCK:
            stream_resume_turn(
                GRAPH,
                thread_id=thread_id,
                user_id=user_id,
                approved=approved,
                reviewer_note=reviewer_note,
                emit_event=emit_event,
            )

    def start_ndjson_stream(self) -> None:
        self._stream_started = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def write_stream_event(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(line)
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the customer-service chat demo server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", default=8000, type=int, help="Port to listen on.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CustomerServiceHandler)
    LOGGER.info("customer_service_server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("customer_service_server stopped by user")
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
