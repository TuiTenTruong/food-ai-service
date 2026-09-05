import json
import time
import requests

MODAL_URL = "https://luanvan--food-ai-service-fastapi-app.modal.run"

def test_live_qwen():
    print(f"[*] 1. Testing Health endpoint at: {MODAL_URL}/health")
    try:
        resp = requests.get(f"{MODAL_URL}/health", timeout=120)
        print(f"    Status code: {resp.status_code}")
        print(f"    Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"    [!] Error: {e}")
        return False

    print("\n[*] 2. Creating Chat Session...")
    try:
        resp = requests.post(f"{MODAL_URL}/api/ai/chat/sessions", json={"title": "Test Qwen Session"}, timeout=30)
        data = resp.json()
        print(f"    Session Response: {data}")
        session_data = data.get("data", {})
        session_id = session_data.get("id") or session_data.get("session_id") or data.get("id")
        if not session_id:
            print("    [!] Failed to get session_id")
            return False
    except Exception as e:
        print(f"    [!] Error creating session: {e}")
        return False

    print(f"\n[*] 3. Sending test message to Qwen 2.5 - 3B on GPU (Session: {session_id})...")
    print("    (Note: First call may take 10-30s if model weights are being loaded into GPU VRAM)")
    start_t = time.time()
    try:
        msg_payload = {
            "message": "Gợi ý cho tôi 1 món ăn ngon từ trứng gà và cà chua, trình bày ngắn gọn các bước."
        }
        resp = requests.post(f"{MODAL_URL}/api/ai/chat/sessions/{session_id}/messages", json=msg_payload, timeout=120)
        elapsed = time.time() - start_t
        print(f"    Response code: {resp.status_code} (took {elapsed:.2f}s)")
        print(f"    Response JSON:\n{json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"    [!] Error sending message: {e}")
        return False

if __name__ == "__main__":
    test_live_qwen()
