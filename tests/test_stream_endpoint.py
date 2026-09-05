import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use gemini for fast local SSE verification
os.environ["LLM_PROVIDER"] = "gemini"

from fastapi.testclient import TestClient
from run_ai import app

def test_stream_api():
    print("\n[*] Testing SSE Streaming Endpoint on FastAPI...")
    client = TestClient(app)

    # 1. Create chat session
    res = client.post("/api/ai/chat/sessions", json={"title": "Test SSE Stream"})
    assert res.status_code == 200, f"Failed to create session: {res.text}"
    session_id = res.json()["data"]["id"]
    print(f"    Created Session ID: {session_id}")

    # 2. Test SSE endpoint with mock/stream
    # Note: On local CPU without Qwen weights, let's verify route exists and accepts request
    req_payload = {
        "message": "Xin chào! Bạn là ai?"
    }

    print(f"    Sending streaming request to /api/ai/chat/sessions/{session_id}/messages/stream...")
    with client.stream("POST", f"/api/ai/chat/sessions/{session_id}/messages/stream", json=req_payload) as response:
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "text/event-stream" in response.headers.get("content-type", "")
        print(f"    Response Status: {response.status_code}")
        print(f"    Content-Type: {response.headers.get('content-type')}")
        
        # Read first few lines of the stream
        lines = []
        for line in response.iter_lines():
            if line:
                lines.append(line)
                if len(lines) >= 3:
                    break
        print(f"    First stream lines received: {lines}")

    print("\n[OK] SSE Streaming Endpoint Verified Successfully!")

if __name__ == "__main__":
    test_stream_api()
